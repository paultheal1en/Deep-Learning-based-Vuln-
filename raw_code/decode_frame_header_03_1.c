static int decode_frame_header(VP8Context *s, const uint8_t *buf, int buf_size)

{

    VP56RangeCoder *c = &s->c;

    int header_size, hscale, vscale, i, j, k, l, m, ret;

    int width  = s->avctx->width;

    int height = s->avctx->height;



    s->keyframe  = !(buf[0] & 1);

    s->profile   =  (buf[0]>>1) & 7;

    s->invisible = !(buf[0] & 0x10);

    header_size  = AV_RL24(buf) >> 5;

    buf      += 3;

    buf_size -= 3;



    if (s->profile > 3)

        av_log(s->avctx, AV_LOG_WARNING, "Unknown profile %d\n", s->profile);



    if (!s->profile)

        memcpy(s->put_pixels_tab, s->vp8dsp.put_vp8_epel_pixels_tab, sizeof(s->put_pixels_tab));

    else    // profile 1-3 use bilinear, 4+ aren't defined so whatever

        memcpy(s->put_pixels_tab, s->vp8dsp.put_vp8_bilinear_pixels_tab, sizeof(s->put_pixels_tab));



    if (header_size > buf_size - 7*s->keyframe) {

        av_log(s->avctx, AV_LOG_ERROR, "Header size larger than data provided\n");

        return AVERROR_INVALIDDATA;

    }



    if (s->keyframe) {

        if (AV_RL24(buf) != 0x2a019d) {

            av_log(s->avctx, AV_LOG_ERROR, "Invalid start code 0x%x\n", AV_RL24(buf));

            return AVERROR_INVALIDDATA;

        }

        width  = AV_RL16(buf+3) & 0x3fff;

        height = AV_RL16(buf+5) & 0x3fff;

        hscale = buf[4] >> 6;

        vscale = buf[6] >> 6;

        buf      += 7;

        buf_size -= 7;



        if (hscale || vscale)

            av_log_missing_feature(s->avctx, "Upscaling", 1);



        s->update_golden = s->update_altref = VP56_FRAME_CURRENT;

        for (i = 0; i < 4; i++)

            for (j = 0; j < 16; j++)

                memcpy(s->prob->token[i][j], vp8_token_default_probs[i][vp8_coeff_band[j]],

                       sizeof(s->prob->token[i][j]));

        memcpy(s->prob->pred16x16, vp8_pred16x16_prob_inter, sizeof(s->prob->pred16x16));

        memcpy(s->prob->pred8x8c , vp8_pred8x8c_prob_inter , sizeof(s->prob->pred8x8c));

        memcpy(s->prob->mvc      , vp8_mv_default_prob     , sizeof(s->prob->mvc));

        memset(&s->segmentation, 0, sizeof(s->segmentation));

    }



    if (!s->macroblocks_base || /* first frame */

        width != s->avctx->width || height != s->avctx->height) {

        if ((ret = update_dimensions(s, width, height)) < 0)

            return ret;

    }



    ff_vp56_init_range_decoder(c, buf, header_size);

    buf      += header_size;

    buf_size -= header_size;



    if (s->keyframe) {

        if (vp8_rac_get(c))

            av_log(s->avctx, AV_LOG_WARNING, "Unspecified colorspace\n");

        vp8_rac_get(c); // whether we can skip clamping in dsp functions

    }



    if ((s->segmentation.enabled = vp8_rac_get(c)))

        parse_segment_info(s);

    else

        s->segmentation.update_map = 0; // FIXME: move this to some init function?



    s->filter.simple    = vp8_rac_get(c);

    s->filter.level     = vp8_rac_get_uint(c, 6);

    s->filter.sharpness = vp8_rac_get_uint(c, 3);



    if ((s->lf_delta.enabled = vp8_rac_get(c)))

        if (vp8_rac_get(c))

            update_lf_deltas(s);



    if (setup_partitions(s, buf, buf_size)) {

        av_log(s->avctx, AV_LOG_ERROR, "Invalid partitions\n");

        return AVERROR_INVALIDDATA;

    }



    get_quants(s);



    if (!s->keyframe) {

        update_refs(s);

        s->sign_bias[VP56_FRAME_GOLDEN]               = vp8_rac_get(c);

        s->sign_bias[VP56_FRAME_GOLDEN2 /* altref */] = vp8_rac_get(c);

    }



    // if we aren't saving this frame's probabilities for future frames,

    // make a copy of the current probabilities

    if (!(s->update_probabilities = vp8_rac_get(c)))

        s->prob[1] = s->prob[0];



    s->update_last = s->keyframe || vp8_rac_get(c);



    for (i = 0; i < 4; i++)

        for (j = 0; j < 8; j++)

            for (k = 0; k < 3; k++)

                for (l = 0; l < NUM_DCT_TOKENS-1; l++)

                    if (vp56_rac_get_prob_branchy(c, vp8_token_update_probs[i][j][k][l])) {

                        int prob = vp8_rac_get_uint(c, 8);

                        for (m = 0; vp8_coeff_band_indexes[j][m] >= 0; m++)

                            s->prob->token[i][vp8_coeff_band_indexes[j][m]][k][l] = prob;

                    }



    if ((s->mbskip_enabled = vp8_rac_get(c)))

        s->prob->mbskip = vp8_rac_get_uint(c, 8);



    if (!s->keyframe) {

        s->prob->intra  = vp8_rac_get_uint(c, 8);

        s->prob->last   = vp8_rac_get_uint(c, 8);

        s->prob->golden = vp8_rac_get_uint(c, 8);



        if (vp8_rac_get(c))

            for (i = 0; i < 4; i++)

                s->prob->pred16x16[i] = vp8_rac_get_uint(c, 8);

        if (vp8_rac_get(c))

            for (i = 0; i < 3; i++)

                s->prob->pred8x8c[i]  = vp8_rac_get_uint(c, 8);



        // 17.2 MV probability update

        for (i = 0; i < 2; i++)

            for (j = 0; j < 19; j++)

                if (vp56_rac_get_prob_branchy(c, vp8_mv_update_prob[i][j]))

                    s->prob->mvc[i][j] = vp8_rac_get_nn(c);

    }



    return 0;

}
