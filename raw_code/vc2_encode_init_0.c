static av_cold int vc2_encode_init(AVCodecContext *avctx)

{

    Plane *p;

    SubBand *b;

    int i, j, level, o, shift;

    const AVPixFmtDescriptor *fmt = av_pix_fmt_desc_get(avctx->pix_fmt);

    const int depth = fmt->comp[0].depth;

    VC2EncContext *s = avctx->priv_data;



    s->picture_number = 0;



    /* Total allowed quantization range */

    s->q_ceil    = DIRAC_MAX_QUANT_INDEX;



    s->ver.major = 2;

    s->ver.minor = 0;

    s->profile   = 3;

    s->level     = 3;



    s->base_vf   = -1;

    s->strict_compliance = 1;



    s->q_avg = 0;

    s->slice_max_bytes = 0;

    s->slice_min_bytes = 0;



    /* Mark unknown as progressive */

    s->interlaced = !((avctx->field_order == AV_FIELD_UNKNOWN) ||

                      (avctx->field_order == AV_FIELD_PROGRESSIVE));



    for (i = 0; i < base_video_fmts_len; i++) {

        const VC2BaseVideoFormat *fmt = &base_video_fmts[i];

        if (avctx->pix_fmt != fmt->pix_fmt)

            continue;

        if (avctx->time_base.num != fmt->time_base.num)

            continue;

        if (avctx->time_base.den != fmt->time_base.den)

            continue;

        if (avctx->width != fmt->width)

            continue;

        if (avctx->height != fmt->height)

            continue;

        if (s->interlaced != fmt->interlaced)

            continue;

        s->base_vf = i;

        s->level   = base_video_fmts[i].level;

        break;

    }



    if (s->interlaced)

        av_log(avctx, AV_LOG_WARNING, "Interlacing enabled!\n");



    if ((s->slice_width  & (s->slice_width  - 1)) ||

        (s->slice_height & (s->slice_height - 1))) {

        av_log(avctx, AV_LOG_ERROR, "Slice size is not a power of two!\n");

        return AVERROR_UNKNOWN;

    }



    if ((s->slice_width > avctx->width) ||

        (s->slice_height > avctx->height)) {

        av_log(avctx, AV_LOG_ERROR, "Slice size is bigger than the image!\n");

        return AVERROR_UNKNOWN;

    }



    if (s->base_vf <= 0) {

        if (avctx->strict_std_compliance <= FF_COMPLIANCE_UNOFFICIAL) {

            s->strict_compliance = s->base_vf = 0;

            av_log(avctx, AV_LOG_WARNING, "Disabling strict compliance\n");

        } else {

            av_log(avctx, AV_LOG_WARNING, "Given format does not strictly comply with "

                   "the specifications, please add a -strict -1 flag to use it\n");

            return AVERROR_UNKNOWN;

        }

    } else {

        av_log(avctx, AV_LOG_INFO, "Selected base video format = %i (%s)\n",

               s->base_vf, base_video_fmts[s->base_vf].name);

    }



    /* Chroma subsampling */

    avcodec_get_chroma_sub_sample(avctx->pix_fmt, &s->chroma_x_shift, &s->chroma_y_shift);



    /* Bit depth and color range index */

    if (depth == 8 && avctx->color_range == AVCOL_RANGE_JPEG) {

        s->bpp = 1;

        s->bpp_idx = 1;

        s->diff_offset = 128;

    } else if (depth == 8 && (avctx->color_range == AVCOL_RANGE_MPEG ||

               avctx->color_range == AVCOL_RANGE_UNSPECIFIED)) {

        s->bpp = 1;

        s->bpp_idx = 2;

        s->diff_offset = 128;

    } else if (depth == 10) {

        s->bpp = 2;

        s->bpp_idx = 3;

        s->diff_offset = 512;

    } else {

        s->bpp = 2;

        s->bpp_idx = 4;

        s->diff_offset = 2048;

    }



    /* Planes initialization */

    for (i = 0; i < 3; i++) {

        int w, h;

        p = &s->plane[i];

        p->width      = avctx->width  >> (i ? s->chroma_x_shift : 0);

        p->height     = avctx->height >> (i ? s->chroma_y_shift : 0);

        if (s->interlaced)

            p->height >>= 1;

        p->dwt_width  = w = FFALIGN(p->width,  (1 << s->wavelet_depth));

        p->dwt_height = h = FFALIGN(p->height, (1 << s->wavelet_depth));

        p->coef_stride = FFALIGN(p->dwt_width, 32);

        p->coef_buf = av_malloc(p->coef_stride*p->dwt_height*sizeof(dwtcoef));

        if (!p->coef_buf)

            goto alloc_fail;

        for (level = s->wavelet_depth-1; level >= 0; level--) {

            w = w >> 1;

            h = h >> 1;

            for (o = 0; o < 4; o++) {

                b = &p->band[level][o];

                b->width  = w;

                b->height = h;

                b->stride = p->coef_stride;

                shift = (o > 1)*b->height*b->stride + (o & 1)*b->width;

                b->buf = p->coef_buf + shift;

            }

        }



        /* DWT init */

        if (ff_vc2enc_init_transforms(&s->transform_args[i].t,

                                      s->plane[i].coef_stride,

                                      s->plane[i].dwt_height))

            goto alloc_fail;

    }



    /* Slices */

    s->num_x = s->plane[0].dwt_width/s->slice_width;

    s->num_y = s->plane[0].dwt_height/s->slice_height;



    s->slice_args = av_calloc(s->num_x*s->num_y, sizeof(SliceArgs));

    if (!s->slice_args)

        goto alloc_fail;



    /* Lookup tables */

    s->coef_lut_len = av_malloc(COEF_LUT_TAB*(s->q_ceil+1)*sizeof(*s->coef_lut_len));

    if (!s->coef_lut_len)

        goto alloc_fail;



    s->coef_lut_val = av_malloc(COEF_LUT_TAB*(s->q_ceil+1)*sizeof(*s->coef_lut_val));

    if (!s->coef_lut_val)

        goto alloc_fail;



    for (i = 0; i < s->q_ceil; i++) {

        uint8_t  *len_lut = &s->coef_lut_len[i*COEF_LUT_TAB];

        uint32_t *val_lut = &s->coef_lut_val[i*COEF_LUT_TAB];

        for (j = 0; j < COEF_LUT_TAB; j++) {

            get_vc2_ue_uint(QUANT(j, ff_dirac_qscale_tab[i]),

                            &len_lut[j], &val_lut[j]);

            if (len_lut[j] != 1) {

                len_lut[j] += 1;

                val_lut[j] <<= 1;

            } else {

                val_lut[j] = 1;

            }

        }

    }



    return 0;



alloc_fail:

    vc2_encode_end(avctx);

    av_log(avctx, AV_LOG_ERROR, "Unable to allocate memory!\n");

    return AVERROR(ENOMEM);

}
