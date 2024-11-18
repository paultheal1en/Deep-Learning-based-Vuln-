static int parse_source_parameters(AVDiracSeqHeader *dsh, GetBitContext *gb,
                                   void *log_ctx)
{
    AVRational frame_rate = { 0, 0 };
    unsigned luma_depth = 8, luma_offset = 16;
    int idx;
    int chroma_x_shift, chroma_y_shift;
    /* [DIRAC_STD] 10.3.2 Frame size. frame_size(video_params) */
    /* [DIRAC_STD] custom_dimensions_flag */
    if (get_bits1(gb)) {
        dsh->width  = svq3_get_ue_golomb(gb); /* [DIRAC_STD] FRAME_WIDTH  */
        dsh->height = svq3_get_ue_golomb(gb); /* [DIRAC_STD] FRAME_HEIGHT */
    }
    /* [DIRAC_STD] 10.3.3 Chroma Sampling Format.
     *  chroma_sampling_format(video_params) */
    /* [DIRAC_STD] custom_chroma_format_flag */
    if (get_bits1(gb))
        /* [DIRAC_STD] CHROMA_FORMAT_INDEX */
        dsh->chroma_format = svq3_get_ue_golomb(gb);
    if (dsh->chroma_format > 2U) {
        if (log_ctx)
            av_log(log_ctx, AV_LOG_ERROR, "Unknown chroma format %d\n",
                   dsh->chroma_format);
    }
    /* [DIRAC_STD] 10.3.4 Scan Format. scan_format(video_params) */
    /* [DIRAC_STD] custom_scan_format_flag */
    if (get_bits1(gb))
        /* [DIRAC_STD] SOURCE_SAMPLING */
        dsh->interlaced = svq3_get_ue_golomb(gb);
    if (dsh->interlaced > 1U)
    /* [DIRAC_STD] 10.3.5 Frame Rate. frame_rate(video_params) */
    if (get_bits1(gb)) { /* [DIRAC_STD] custom_frame_rate_flag */
        dsh->frame_rate_index = svq3_get_ue_golomb(gb);
        if (dsh->frame_rate_index > 10U)
        if (!dsh->frame_rate_index) {
            /* [DIRAC_STD] FRAME_RATE_NUMER */
            frame_rate.num = svq3_get_ue_golomb(gb);
            /* [DIRAC_STD] FRAME_RATE_DENOM */
            frame_rate.den = svq3_get_ue_golomb(gb);
        }
    }
    /* [DIRAC_STD] preset_frame_rate(video_params, index) */
    if (dsh->frame_rate_index > 0) {
        if (dsh->frame_rate_index <= 8)
            frame_rate = ff_mpeg12_frame_rate_tab[dsh->frame_rate_index];
        else
            /* [DIRAC_STD] Table 10.3 values 9-10 */
            frame_rate = dirac_frame_rate[dsh->frame_rate_index - 9];
    }
    dsh->framerate = frame_rate;
    /* [DIRAC_STD] 10.3.6 Pixel Aspect Ratio.
     * pixel_aspect_ratio(video_params) */
    if (get_bits1(gb)) { /* [DIRAC_STD] custom_pixel_aspect_ratio_flag */
        /* [DIRAC_STD] index */
        dsh->aspect_ratio_index = svq3_get_ue_golomb(gb);
        if (dsh->aspect_ratio_index > 6U)
        if (!dsh->aspect_ratio_index) {
            dsh->sample_aspect_ratio.num = svq3_get_ue_golomb(gb);
            dsh->sample_aspect_ratio.den = svq3_get_ue_golomb(gb);
        }
    }
    /* [DIRAC_STD] Take value from Table 10.4 Available preset pixel
     *  aspect ratio values */
    if (dsh->aspect_ratio_index > 0)
        dsh->sample_aspect_ratio =
            dirac_preset_aspect_ratios[dsh->aspect_ratio_index - 1];
    /* [DIRAC_STD] 10.3.7 Clean area. clean_area(video_params) */
    if (get_bits1(gb)) { /* [DIRAC_STD] custom_clean_area_flag */
        /* [DIRAC_STD] CLEAN_WIDTH */
        dsh->clean_width = svq3_get_ue_golomb(gb);
        /* [DIRAC_STD] CLEAN_HEIGHT */
        dsh->clean_height = svq3_get_ue_golomb(gb);
        /* [DIRAC_STD] CLEAN_LEFT_OFFSET */
        dsh->clean_left_offset = svq3_get_ue_golomb(gb);
        /* [DIRAC_STD] CLEAN_RIGHT_OFFSET */
        dsh->clean_right_offset = svq3_get_ue_golomb(gb);
    }
    /* [DIRAC_STD] 10.3.8 Signal range. signal_range(video_params)
     * WARNING: Some adaptation seems to be done using the
     * AVCOL_RANGE_MPEG/JPEG values */
    if (get_bits1(gb)) { /* [DIRAC_STD] custom_signal_range_flag */
        /* [DIRAC_STD] index */
        dsh->pixel_range_index = svq3_get_ue_golomb(gb);
        if (dsh->pixel_range_index > 4U)
        /* This assumes either fullrange or MPEG levels only */
        if (!dsh->pixel_range_index) {
            luma_offset = svq3_get_ue_golomb(gb);
            luma_depth  = av_log2(svq3_get_ue_golomb(gb)) + 1;
            svq3_get_ue_golomb(gb); /* chroma offset    */
            svq3_get_ue_golomb(gb); /* chroma excursion */
            dsh->color_range = luma_offset ? AVCOL_RANGE_MPEG
                                           : AVCOL_RANGE_JPEG;
        }
    }
    /* [DIRAC_STD] Table 10.5
     * Available signal range presets <--> pixel_range_presets */
    if (dsh->pixel_range_index > 0) {
        idx                = dsh->pixel_range_index - 1;
        luma_depth         = pixel_range_presets[idx].bitdepth;
        dsh->color_range   = pixel_range_presets[idx].color_range;
    }
    dsh->bit_depth = luma_depth;
    dsh->pix_fmt = dirac_pix_fmt[dsh->chroma_format][dsh->pixel_range_index-2];
    avcodec_get_chroma_sub_sample(dsh->pix_fmt, &chroma_x_shift, &chroma_y_shift);
    if ((dsh->width % (1<<chroma_x_shift)) || (dsh->height % (1<<chroma_y_shift))) {
        if (log_ctx)
            av_log(log_ctx, AV_LOG_ERROR, "Dimensions must be an integer multiple of the chroma subsampling\n");
    }
    /* [DIRAC_STD] 10.3.9 Colour specification. colour_spec(video_params) */
    if (get_bits1(gb)) { /* [DIRAC_STD] custom_colour_spec_flag */
        /* [DIRAC_STD] index */
        idx = dsh->color_spec_index = svq3_get_ue_golomb(gb);
        if (dsh->color_spec_index > 4U)
        dsh->color_primaries = dirac_color_presets[idx].color_primaries;
        dsh->colorspace      = dirac_color_presets[idx].colorspace;
        dsh->color_trc       = dirac_color_presets[idx].color_trc;
        if (!dsh->color_spec_index) {
            /* [DIRAC_STD] 10.3.9.1 Colour primaries */
            if (get_bits1(gb)) {
                idx = svq3_get_ue_golomb(gb);
                if (idx < 3U)
                    dsh->color_primaries = dirac_primaries[idx];
            }
            /* [DIRAC_STD] 10.3.9.2 Colour matrix */
            if (get_bits1(gb)) {
                idx = svq3_get_ue_golomb(gb);
                if (!idx)
                    dsh->colorspace = AVCOL_SPC_BT709;
                else if (idx == 1)
                    dsh->colorspace = AVCOL_SPC_BT470BG;
            }
            /* [DIRAC_STD] 10.3.9.3 Transfer function */
            if (get_bits1(gb) && !svq3_get_ue_golomb(gb))
                dsh->color_trc = AVCOL_TRC_BT709;
        }
    } else {
        idx                    = dsh->color_spec_index;
        dsh->color_primaries = dirac_color_presets[idx].color_primaries;
        dsh->colorspace      = dirac_color_presets[idx].colorspace;
        dsh->color_trc       = dirac_color_presets[idx].color_trc;
    }
    return 0;
}