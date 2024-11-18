int ff_h264_decode_slice_header(H264Context *h, H264SliceContext *sl)

{

    const SPS *sps;

    const PPS *pps;

    unsigned int first_mb_in_slice;

    unsigned int pps_id;

    int ret;

    unsigned int slice_type, tmp, i, j;

    int last_pic_structure, last_pic_droppable;

    int needs_reinit = 0;

    int field_pic_flag, bottom_field_flag;

    int frame_num, droppable, picture_structure;

    int mb_aff_frame = 0;



    h->qpel_put = h->h264qpel.put_h264_qpel_pixels_tab;

    h->qpel_avg = h->h264qpel.avg_h264_qpel_pixels_tab;



    first_mb_in_slice = get_ue_golomb(&sl->gb);



    if (first_mb_in_slice == 0) { // FIXME better field boundary detection

        if (h->current_slice && h->cur_pic_ptr && FIELD_PICTURE(h)) {

            ff_h264_field_end(h, sl, 1);

        }



        h->current_slice = 0;

        if (!h->first_field) {

            if (h->cur_pic_ptr && !h->droppable) {

                ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,

                                          h->picture_structure == PICT_BOTTOM_FIELD);

            }

            h->cur_pic_ptr = NULL;

        }

    }



    slice_type = get_ue_golomb_31(&sl->gb);

    if (slice_type > 9) {

        av_log(h->avctx, AV_LOG_ERROR,

               "slice type %d too large at %d\n",

               slice_type, first_mb_in_slice);

        return AVERROR_INVALIDDATA;

    }

    if (slice_type > 4) {

        slice_type -= 5;

        sl->slice_type_fixed = 1;

    } else

        sl->slice_type_fixed = 0;



    slice_type         = ff_h264_golomb_to_pict_type[slice_type];

    sl->slice_type     = slice_type;

    sl->slice_type_nos = slice_type & 3;



    if (h->nal_unit_type  == NAL_IDR_SLICE &&

        sl->slice_type_nos != AV_PICTURE_TYPE_I) {

        av_log(h->avctx, AV_LOG_ERROR, "A non-intra slice in an IDR NAL unit.\n");

        return AVERROR_INVALIDDATA;

    }



    // to make a few old functions happy, it's wrong though

    if (!h->setup_finished)

        h->pict_type = sl->slice_type;



    pps_id = get_ue_golomb(&sl->gb);

    if (pps_id >= MAX_PPS_COUNT) {

        av_log(h->avctx, AV_LOG_ERROR, "pps_id %u out of range\n", pps_id);

        return AVERROR_INVALIDDATA;

    }

    if (!h->ps.pps_list[pps_id]) {

        av_log(h->avctx, AV_LOG_ERROR,

               "non-existing PPS %u referenced\n",

               pps_id);

        return AVERROR_INVALIDDATA;

    }

    if (!h->setup_finished) {

        h->ps.pps = (const PPS*)h->ps.pps_list[pps_id]->data;

    } else if (h->ps.pps != (const PPS*)h->ps.pps_list[pps_id]->data) {

        av_log(h->avctx, AV_LOG_ERROR, "PPS changed between slices\n");

        return AVERROR_INVALIDDATA;

    }



    if (!h->ps.sps_list[h->ps.pps->sps_id]) {

        av_log(h->avctx, AV_LOG_ERROR,

               "non-existing SPS %u referenced\n",

               h->ps.pps->sps_id);

        return AVERROR_INVALIDDATA;

    }



    if (h->ps.sps != (const SPS*)h->ps.sps_list[h->ps.pps->sps_id]->data) {

        h->ps.sps = (SPS*)h->ps.sps_list[h->ps.pps->sps_id]->data;



        if (h->bit_depth_luma    != h->ps.sps->bit_depth_luma ||

            h->chroma_format_idc != h->ps.sps->chroma_format_idc)

            needs_reinit         = 1;



        if (h->flags & AV_CODEC_FLAG_LOW_DELAY ||

            (h->ps.sps->bitstream_restriction_flag &&

             !h->ps.sps->num_reorder_frames)) {

            if (h->avctx->has_b_frames > 1 || h->delayed_pic[0])

                av_log(h->avctx, AV_LOG_WARNING, "Delayed frames seen. "

                       "Reenabling low delay requires a codec flush.\n");

            else

                h->low_delay = 1;

        }



        if (h->avctx->has_b_frames < 2)

            h->avctx->has_b_frames = !h->low_delay;



    }



    pps = h->ps.pps;

    sps = h->ps.sps;



    if (!h->setup_finished) {

        h->avctx->profile = ff_h264_get_profile(sps);

        h->avctx->level   = sps->level_idc;

        h->avctx->refs    = sps->ref_frame_count;



        if (h->mb_width  != sps->mb_width ||

            h->mb_height != sps->mb_height * (2 - sps->frame_mbs_only_flag))

            needs_reinit = 1;



        h->mb_width  = sps->mb_width;

        h->mb_height = sps->mb_height * (2 - sps->frame_mbs_only_flag);

        h->mb_num    = h->mb_width * h->mb_height;

        h->mb_stride = h->mb_width + 1;



        h->b_stride = h->mb_width * 4;



        h->chroma_y_shift = sps->chroma_format_idc <= 1; // 400 uses yuv420p



        h->width  = 16 * h->mb_width;

        h->height = 16 * h->mb_height;



        ret = init_dimensions(h);

        if (ret < 0)

            return ret;



        if (sps->video_signal_type_present_flag) {

            h->avctx->color_range = sps->full_range ? AVCOL_RANGE_JPEG

                : AVCOL_RANGE_MPEG;

            if (sps->colour_description_present_flag) {

                if (h->avctx->colorspace != sps->colorspace)

                    needs_reinit = 1;

                h->avctx->color_primaries = sps->color_primaries;

                h->avctx->color_trc       = sps->color_trc;

                h->avctx->colorspace      = sps->colorspace;

            }

        }

    }



    if (h->context_initialized && needs_reinit) {

        h->context_initialized = 0;

        if (sl != h->slice_ctx) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "changing width %d -> %d / height %d -> %d on "

                   "slice %d\n",

                   h->width, h->avctx->coded_width,

                   h->height, h->avctx->coded_height,

                   h->current_slice + 1);

            return AVERROR_INVALIDDATA;

        }



        ff_h264_flush_change(h);



        if ((ret = get_pixel_format(h)) < 0)

            return ret;

        h->avctx->pix_fmt = ret;



        av_log(h->avctx, AV_LOG_INFO, "Reinit context to %dx%d, "

               "pix_fmt: %d\n", h->width, h->height, h->avctx->pix_fmt);



        if ((ret = h264_slice_header_init(h)) < 0) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "h264_slice_header_init() failed\n");

            return ret;

        }

    }

    if (!h->context_initialized) {

        if (sl != h->slice_ctx) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "Cannot (re-)initialize context during parallel decoding.\n");

            return AVERROR_PATCHWELCOME;

        }



        if ((ret = get_pixel_format(h)) < 0)

            return ret;

        h->avctx->pix_fmt = ret;



        if ((ret = h264_slice_header_init(h)) < 0) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "h264_slice_header_init() failed\n");

            return ret;

        }

    }



    frame_num = get_bits(&sl->gb, sps->log2_max_frame_num);

    if (!h->setup_finished)

        h->frame_num = frame_num;



    sl->mb_mbaff       = 0;



    last_pic_structure = h->picture_structure;

    last_pic_droppable = h->droppable;



    droppable = h->nal_ref_idc == 0;

    if (sps->frame_mbs_only_flag) {

        picture_structure = PICT_FRAME;

    } else {

        field_pic_flag = get_bits1(&sl->gb);

        if (field_pic_flag) {

            bottom_field_flag = get_bits1(&sl->gb);

            picture_structure = PICT_TOP_FIELD + bottom_field_flag;

        } else {

            picture_structure = PICT_FRAME;

            mb_aff_frame      = sps->mb_aff;

        }

    }

    if (!h->setup_finished) {

        h->droppable         = droppable;

        h->picture_structure = picture_structure;

        h->mb_aff_frame      = mb_aff_frame;

    }

    sl->mb_field_decoding_flag = h->picture_structure != PICT_FRAME;



    if (h->current_slice != 0) {

        if (last_pic_structure != picture_structure ||

            last_pic_droppable != droppable) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "Changing field mode (%d -> %d) between slices is not allowed\n",

                   last_pic_structure, h->picture_structure);

            return AVERROR_INVALIDDATA;

        } else if (!h->cur_pic_ptr) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "unset cur_pic_ptr on slice %d\n",

                   h->current_slice + 1);

            return AVERROR_INVALIDDATA;

        }

    } else {

        /* Shorten frame num gaps so we don't have to allocate reference

         * frames just to throw them away */

        if (h->frame_num != h->prev_frame_num) {

            int unwrap_prev_frame_num = h->prev_frame_num;

            int max_frame_num         = 1 << sps->log2_max_frame_num;



            if (unwrap_prev_frame_num > h->frame_num)

                unwrap_prev_frame_num -= max_frame_num;



            if ((h->frame_num - unwrap_prev_frame_num) > sps->ref_frame_count) {

                unwrap_prev_frame_num = (h->frame_num - sps->ref_frame_count) - 1;

                if (unwrap_prev_frame_num < 0)

                    unwrap_prev_frame_num += max_frame_num;



                h->prev_frame_num = unwrap_prev_frame_num;

            }

        }



        /* See if we have a decoded first field looking for a pair...

         * Here, we're using that to see if we should mark previously

         * decode frames as "finished".

         * We have to do that before the "dummy" in-between frame allocation,

         * since that can modify s->current_picture_ptr. */

        if (h->first_field) {

            assert(h->cur_pic_ptr);

            assert(h->cur_pic_ptr->f->buf[0]);

            assert(h->cur_pic_ptr->reference != DELAYED_PIC_REF);



            /* figure out if we have a complementary field pair */

            if (!FIELD_PICTURE(h) || h->picture_structure == last_pic_structure) {

                /* Previous field is unmatched. Don't display it, but let it

                 * remain for reference if marked as such. */

                if (!last_pic_droppable && last_pic_structure != PICT_FRAME) {

                    ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,

                                              last_pic_structure == PICT_TOP_FIELD);

                }

            } else {

                if (h->cur_pic_ptr->frame_num != h->frame_num) {

                    /* This and previous field were reference, but had

                     * different frame_nums. Consider this field first in

                     * pair. Throw away previous field except for reference

                     * purposes. */

                    if (!last_pic_droppable && last_pic_structure != PICT_FRAME) {

                        ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX,

                                                  last_pic_structure == PICT_TOP_FIELD);

                    }

                } else {

                    /* Second field in complementary pair */

                    if (!((last_pic_structure   == PICT_TOP_FIELD &&

                           h->picture_structure == PICT_BOTTOM_FIELD) ||

                          (last_pic_structure   == PICT_BOTTOM_FIELD &&

                           h->picture_structure == PICT_TOP_FIELD))) {

                        av_log(h->avctx, AV_LOG_ERROR,

                               "Invalid field mode combination %d/%d\n",

                               last_pic_structure, h->picture_structure);

                        h->picture_structure = last_pic_structure;

                        h->droppable         = last_pic_droppable;

                        return AVERROR_INVALIDDATA;

                    } else if (last_pic_droppable != h->droppable) {

                        avpriv_request_sample(h->avctx,

                                              "Found reference and non-reference fields in the same frame, which");

                        h->picture_structure = last_pic_structure;

                        h->droppable         = last_pic_droppable;

                        return AVERROR_PATCHWELCOME;

                    }

                }

            }

        }



        while (h->frame_num != h->prev_frame_num &&

               h->frame_num != (h->prev_frame_num + 1) % (1 << sps->log2_max_frame_num)) {

            H264Picture *prev = h->short_ref_count ? h->short_ref[0] : NULL;

            av_log(h->avctx, AV_LOG_DEBUG, "Frame num gap %d %d\n",

                   h->frame_num, h->prev_frame_num);

            ret = initialize_cur_frame(h);

            if (ret < 0) {

                h->first_field = 0;

                return ret;

            }



            h->prev_frame_num++;

            h->prev_frame_num        %= 1 << sps->log2_max_frame_num;

            h->cur_pic_ptr->frame_num = h->prev_frame_num;

            ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX, 0);

            ff_thread_report_progress(&h->cur_pic_ptr->tf, INT_MAX, 1);

            ret = ff_generate_sliding_window_mmcos(h, 1);

            if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))

                return ret;

            ret = ff_h264_execute_ref_pic_marking(h, h->mmco, h->mmco_index);

            if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))

                return ret;

            /* Error concealment: If a ref is missing, copy the previous ref

             * in its place.

             * FIXME: Avoiding a memcpy would be nice, but ref handling makes

             * many assumptions about there being no actual duplicates.

             * FIXME: This does not copy padding for out-of-frame motion

             * vectors.  Given we are concealing a lost frame, this probably

             * is not noticeable by comparison, but it should be fixed. */

            if (h->short_ref_count) {

                if (prev &&

                    h->short_ref[0]->f->width == prev->f->width &&

                    h->short_ref[0]->f->height == prev->f->height &&

                    h->short_ref[0]->f->format == prev->f->format) {

                    av_image_copy(h->short_ref[0]->f->data,

                                  h->short_ref[0]->f->linesize,

                                  (const uint8_t **)prev->f->data,

                                  prev->f->linesize,

                                  prev->f->format,

                                  h->mb_width  * 16,

                                  h->mb_height * 16);

                    h->short_ref[0]->poc = prev->poc + 2;

                }

                h->short_ref[0]->frame_num = h->prev_frame_num;

            }

        }



        /* See if we have a decoded first field looking for a pair...

         * We're using that to see whether to continue decoding in that

         * frame, or to allocate a new one. */

        if (h->first_field) {

            assert(h->cur_pic_ptr);

            assert(h->cur_pic_ptr->f->buf[0]);

            assert(h->cur_pic_ptr->reference != DELAYED_PIC_REF);



            /* figure out if we have a complementary field pair */

            if (!FIELD_PICTURE(h) || h->picture_structure == last_pic_structure) {

                /* Previous field is unmatched. Don't display it, but let it

                 * remain for reference if marked as such. */

                h->cur_pic_ptr = NULL;

                h->first_field = FIELD_PICTURE(h);

            } else {

                if (h->cur_pic_ptr->frame_num != h->frame_num) {

                    /* This and the previous field had different frame_nums.

                     * Consider this field first in pair. Throw away previous

                     * one except for reference purposes. */

                    h->first_field = 1;

                    h->cur_pic_ptr = NULL;

                } else {

                    /* Second field in complementary pair */

                    h->first_field = 0;

                }

            }

        } else {

            /* Frame or first field in a potentially complementary pair */

            h->first_field = FIELD_PICTURE(h);

        }



        if (!FIELD_PICTURE(h) || h->first_field) {

            if (h264_frame_start(h) < 0) {

                h->first_field = 0;

                return AVERROR_INVALIDDATA;

            }

        } else {

            release_unused_pictures(h, 0);

        }

    }



    assert(h->mb_num == h->mb_width * h->mb_height);

    if (first_mb_in_slice << FIELD_OR_MBAFF_PICTURE(h) >= h->mb_num ||

        first_mb_in_slice >= h->mb_num) {

        av_log(h->avctx, AV_LOG_ERROR, "first_mb_in_slice overflow\n");

        return AVERROR_INVALIDDATA;

    }

    sl->resync_mb_x = sl->mb_x =  first_mb_in_slice % h->mb_width;

    sl->resync_mb_y = sl->mb_y = (first_mb_in_slice / h->mb_width) <<

                                 FIELD_OR_MBAFF_PICTURE(h);

    if (h->picture_structure == PICT_BOTTOM_FIELD)

        sl->resync_mb_y = sl->mb_y = sl->mb_y + 1;

    assert(sl->mb_y < h->mb_height);



    if (h->picture_structure == PICT_FRAME) {

        h->curr_pic_num = h->frame_num;

        h->max_pic_num  = 1 << sps->log2_max_frame_num;

    } else {

        h->curr_pic_num = 2 * h->frame_num + 1;

        h->max_pic_num  = 1 << (sps->log2_max_frame_num + 1);

    }



    if (h->nal_unit_type == NAL_IDR_SLICE)

        get_ue_golomb(&sl->gb); /* idr_pic_id */



    if (sps->poc_type == 0) {

        int poc_lsb = get_bits(&sl->gb, sps->log2_max_poc_lsb);



        if (!h->setup_finished)

            h->poc_lsb = poc_lsb;



        if (pps->pic_order_present == 1 && h->picture_structure == PICT_FRAME) {

            int delta_poc_bottom = get_se_golomb(&sl->gb);

            if (!h->setup_finished)

                h->delta_poc_bottom = delta_poc_bottom;

        }

    }



    if (sps->poc_type == 1 && !sps->delta_pic_order_always_zero_flag) {

        int delta_poc = get_se_golomb(&sl->gb);



        if (!h->setup_finished)

            h->delta_poc[0] = delta_poc;



        if (pps->pic_order_present == 1 && h->picture_structure == PICT_FRAME) {

            delta_poc = get_se_golomb(&sl->gb);



            if (!h->setup_finished)

                h->delta_poc[1] = delta_poc;

        }

    }



    if (!h->setup_finished)

        ff_init_poc(h, h->cur_pic_ptr->field_poc, &h->cur_pic_ptr->poc);



    if (pps->redundant_pic_cnt_present)

        sl->redundant_pic_count = get_ue_golomb(&sl->gb);



    if (sl->slice_type_nos == AV_PICTURE_TYPE_B)

        sl->direct_spatial_mv_pred = get_bits1(&sl->gb);



    ret = ff_h264_parse_ref_count(&sl->list_count, sl->ref_count,

                                  &sl->gb, pps, sl->slice_type_nos,

                                  h->picture_structure);

    if (ret < 0)

        return ret;



    if (sl->slice_type_nos != AV_PICTURE_TYPE_I) {

       ret = ff_h264_decode_ref_pic_list_reordering(h, sl);

       if (ret < 0) {

           sl->ref_count[1] = sl->ref_count[0] = 0;

           return ret;

       }

    }



    if ((pps->weighted_pred && sl->slice_type_nos == AV_PICTURE_TYPE_P) ||

        (pps->weighted_bipred_idc == 1 &&

         sl->slice_type_nos == AV_PICTURE_TYPE_B))

        ff_h264_pred_weight_table(&sl->gb, sps, sl->ref_count,

                                  sl->slice_type_nos, &sl->pwt);

    else if (pps->weighted_bipred_idc == 2 &&

             sl->slice_type_nos == AV_PICTURE_TYPE_B) {

        implicit_weight_table(h, sl, -1);

    } else {

        sl->pwt.use_weight = 0;

        for (i = 0; i < 2; i++) {

            sl->pwt.luma_weight_flag[i]   = 0;

            sl->pwt.chroma_weight_flag[i] = 0;

        }

    }



    // If frame-mt is enabled, only update mmco tables for the first slice

    // in a field. Subsequent slices can temporarily clobber h->mmco_index

    // or h->mmco, which will cause ref list mix-ups and decoding errors

    // further down the line. This may break decoding if the first slice is

    // corrupt, thus we only do this if frame-mt is enabled.

    if (h->nal_ref_idc) {

        ret = ff_h264_decode_ref_pic_marking(h, &sl->gb,

                                             !(h->avctx->active_thread_type & FF_THREAD_FRAME) ||

                                             h->current_slice == 0);

        if (ret < 0 && (h->avctx->err_recognition & AV_EF_EXPLODE))

            return AVERROR_INVALIDDATA;

    }



    if (FRAME_MBAFF(h)) {

        ff_h264_fill_mbaff_ref_list(h, sl);



        if (pps->weighted_bipred_idc == 2 && sl->slice_type_nos == AV_PICTURE_TYPE_B) {

            implicit_weight_table(h, sl, 0);

            implicit_weight_table(h, sl, 1);

        }

    }



    if (sl->slice_type_nos == AV_PICTURE_TYPE_B && !sl->direct_spatial_mv_pred)

        ff_h264_direct_dist_scale_factor(h, sl);

    ff_h264_direct_ref_list_init(h, sl);



    if (sl->slice_type_nos != AV_PICTURE_TYPE_I && pps->cabac) {

        tmp = get_ue_golomb_31(&sl->gb);

        if (tmp > 2) {

            av_log(h->avctx, AV_LOG_ERROR, "cabac_init_idc %u overflow\n", tmp);

            return AVERROR_INVALIDDATA;

        }

        sl->cabac_init_idc = tmp;

    }



    sl->last_qscale_diff = 0;

    tmp = pps->init_qp + get_se_golomb(&sl->gb);

    if (tmp > 51 + 6 * (sps->bit_depth_luma - 8)) {

        av_log(h->avctx, AV_LOG_ERROR, "QP %u out of range\n", tmp);

        return AVERROR_INVALIDDATA;

    }

    sl->qscale       = tmp;

    sl->chroma_qp[0] = get_chroma_qp(h, 0, sl->qscale);

    sl->chroma_qp[1] = get_chroma_qp(h, 1, sl->qscale);

    // FIXME qscale / qp ... stuff

    if (sl->slice_type == AV_PICTURE_TYPE_SP)

        get_bits1(&sl->gb); /* sp_for_switch_flag */

    if (sl->slice_type == AV_PICTURE_TYPE_SP ||

        sl->slice_type == AV_PICTURE_TYPE_SI)

        get_se_golomb(&sl->gb); /* slice_qs_delta */



    sl->deblocking_filter     = 1;

    sl->slice_alpha_c0_offset = 0;

    sl->slice_beta_offset     = 0;

    if (pps->deblocking_filter_parameters_present) {

        tmp = get_ue_golomb_31(&sl->gb);

        if (tmp > 2) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "deblocking_filter_idc %u out of range\n", tmp);

            return AVERROR_INVALIDDATA;

        }

        sl->deblocking_filter = tmp;

        if (sl->deblocking_filter < 2)

            sl->deblocking_filter ^= 1;  // 1<->0



        if (sl->deblocking_filter) {

            sl->slice_alpha_c0_offset = get_se_golomb(&sl->gb) * 2;

            sl->slice_beta_offset     = get_se_golomb(&sl->gb) * 2;

            if (sl->slice_alpha_c0_offset >  12 ||

                sl->slice_alpha_c0_offset < -12 ||

                sl->slice_beta_offset >  12     ||

                sl->slice_beta_offset < -12) {

                av_log(h->avctx, AV_LOG_ERROR,

                       "deblocking filter parameters %d %d out of range\n",

                       sl->slice_alpha_c0_offset, sl->slice_beta_offset);

                return AVERROR_INVALIDDATA;

            }

        }

    }



    if (h->avctx->skip_loop_filter >= AVDISCARD_ALL ||

        (h->avctx->skip_loop_filter >= AVDISCARD_NONKEY &&

         sl->slice_type_nos != AV_PICTURE_TYPE_I) ||

        (h->avctx->skip_loop_filter >= AVDISCARD_BIDIR  &&

         sl->slice_type_nos == AV_PICTURE_TYPE_B) ||

        (h->avctx->skip_loop_filter >= AVDISCARD_NONREF &&

         h->nal_ref_idc == 0))

        sl->deblocking_filter = 0;



    if (sl->deblocking_filter == 1 && h->max_contexts > 1) {

        if (h->avctx->flags2 & AV_CODEC_FLAG2_FAST) {

            /* Cheat slightly for speed:

             * Do not bother to deblock across slices. */

            sl->deblocking_filter = 2;

        } else {

            h->max_contexts = 1;

            if (!h->single_decode_warning) {

                av_log(h->avctx, AV_LOG_INFO,

                       "Cannot parallelize deblocking type 1, decoding such frames in sequential order\n");

                h->single_decode_warning = 1;

            }

            if (sl != h->slice_ctx) {

                av_log(h->avctx, AV_LOG_ERROR,

                       "Deblocking switched inside frame.\n");

                return 1;

            }

        }

    }

    sl->qp_thresh = 15 -

                   FFMIN(sl->slice_alpha_c0_offset, sl->slice_beta_offset) -

                   FFMAX3(0,

                          pps->chroma_qp_index_offset[0],

                          pps->chroma_qp_index_offset[1]) +

                   6 * (sps->bit_depth_luma - 8);



    sl->slice_num       = ++h->current_slice;

    if (sl->slice_num >= MAX_SLICES) {

        av_log(h->avctx, AV_LOG_ERROR,

               "Too many slices, increase MAX_SLICES and recompile\n");

    }



    for (j = 0; j < 2; j++) {

        int id_list[16];

        int *ref2frm = sl->ref2frm[sl->slice_num & (MAX_SLICES - 1)][j];

        for (i = 0; i < 16; i++) {

            id_list[i] = 60;

            if (j < sl->list_count && i < sl->ref_count[j] &&

                sl->ref_list[j][i].parent->f->buf[0]) {

                int k;

                AVBuffer *buf = sl->ref_list[j][i].parent->f->buf[0]->buffer;

                for (k = 0; k < h->short_ref_count; k++)

                    if (h->short_ref[k]->f->buf[0]->buffer == buf) {

                        id_list[i] = k;

                        break;

                    }

                for (k = 0; k < h->long_ref_count; k++)

                    if (h->long_ref[k] && h->long_ref[k]->f->buf[0]->buffer == buf) {

                        id_list[i] = h->short_ref_count + k;

                        break;

                    }

            }

        }



        ref2frm[0] =

        ref2frm[1] = -1;

        for (i = 0; i < 16; i++)

            ref2frm[i + 2] = 4 * id_list[i] + (sl->ref_list[j][i].reference & 3);

        ref2frm[18 + 0] =

        ref2frm[18 + 1] = -1;

        for (i = 16; i < 48; i++)

            ref2frm[i + 4] = 4 * id_list[(i - 16) >> 1] +

                             (sl->ref_list[j][i].reference & 3);

    }



    if (h->avctx->debug & FF_DEBUG_PICT_INFO) {

        av_log(h->avctx, AV_LOG_DEBUG,

               "slice:%d %s mb:%d %c%s%s pps:%u frame:%d poc:%d/%d ref:%d/%d qp:%d loop:%d:%d:%d weight:%d%s %s\n",

               sl->slice_num,

               (h->picture_structure == PICT_FRAME ? "F" : h->picture_structure == PICT_TOP_FIELD ? "T" : "B"),

               first_mb_in_slice,

               av_get_picture_type_char(sl->slice_type),

               sl->slice_type_fixed ? " fix" : "",

               h->nal_unit_type == NAL_IDR_SLICE ? " IDR" : "",

               pps_id, h->frame_num,

               h->cur_pic_ptr->field_poc[0],

               h->cur_pic_ptr->field_poc[1],

               sl->ref_count[0], sl->ref_count[1],

               sl->qscale,

               sl->deblocking_filter,

               sl->slice_alpha_c0_offset, sl->slice_beta_offset,

               sl->pwt.use_weight,

               sl->pwt.use_weight == 1 && sl->pwt.use_weight_chroma ? "c" : "",

               sl->slice_type == AV_PICTURE_TYPE_B ? (sl->direct_spatial_mv_pred ? "SPAT" : "TEMP") : "");

    }



    return 0;

}
