static int hls_coding_unit(HEVCContext *s, int x0, int y0, int log2_cb_size)

{

    int cb_size          = 1 << log2_cb_size;

    HEVCLocalContext *lc = &s->HEVClc;

    int log2_min_cb_size = s->sps->log2_min_cb_size;

    int length           = cb_size >> log2_min_cb_size;

    int min_cb_width     = s->sps->min_cb_width;

    int x_cb             = x0 >> log2_min_cb_size;

    int y_cb             = y0 >> log2_min_cb_size;

    int x, y;



    lc->cu.x                = x0;

    lc->cu.y                = y0;

    lc->cu.rqt_root_cbf     = 1;

    lc->cu.pred_mode        = MODE_INTRA;

    lc->cu.part_mode        = PART_2Nx2N;

    lc->cu.intra_split_flag = 0;

    lc->cu.pcm_flag         = 0;



    SAMPLE_CTB(s->skip_flag, x_cb, y_cb) = 0;

    for (x = 0; x < 4; x++)

        lc->pu.intra_pred_mode[x] = 1;

    if (s->pps->transquant_bypass_enable_flag) {

        lc->cu.cu_transquant_bypass_flag = ff_hevc_cu_transquant_bypass_flag_decode(s);

        if (lc->cu.cu_transquant_bypass_flag)

            set_deblocking_bypass(s, x0, y0, log2_cb_size);

    } else

        lc->cu.cu_transquant_bypass_flag = 0;



    if (s->sh.slice_type != I_SLICE) {

        uint8_t skip_flag = ff_hevc_skip_flag_decode(s, x0, y0, x_cb, y_cb);



        lc->cu.pred_mode = MODE_SKIP;

        x = y_cb * min_cb_width + x_cb;

        for (y = 0; y < length; y++) {

            memset(&s->skip_flag[x], skip_flag, length);

            x += min_cb_width;

        }

        lc->cu.pred_mode = skip_flag ? MODE_SKIP : MODE_INTER;

    }



    if (SAMPLE_CTB(s->skip_flag, x_cb, y_cb)) {

        hls_prediction_unit(s, x0, y0, cb_size, cb_size, log2_cb_size, 0);

        intra_prediction_unit_default_value(s, x0, y0, log2_cb_size);



        if (!s->sh.disable_deblocking_filter_flag)

            ff_hevc_deblocking_boundary_strengths(s, x0, y0, log2_cb_size,

                                                  lc->slice_or_tiles_up_boundary,

                                                  lc->slice_or_tiles_left_boundary);

    } else {

        if (s->sh.slice_type != I_SLICE)

            lc->cu.pred_mode = ff_hevc_pred_mode_decode(s);

        if (lc->cu.pred_mode != MODE_INTRA ||

            log2_cb_size == s->sps->log2_min_cb_size) {

            lc->cu.part_mode        = ff_hevc_part_mode_decode(s, log2_cb_size);

            lc->cu.intra_split_flag = lc->cu.part_mode == PART_NxN &&

                                      lc->cu.pred_mode == MODE_INTRA;

        }



        if (lc->cu.pred_mode == MODE_INTRA) {

            if (lc->cu.part_mode == PART_2Nx2N && s->sps->pcm_enabled_flag &&

                log2_cb_size >= s->sps->pcm.log2_min_pcm_cb_size &&

                log2_cb_size <= s->sps->pcm.log2_max_pcm_cb_size) {

                lc->cu.pcm_flag = ff_hevc_pcm_flag_decode(s);

            }

            if (lc->cu.pcm_flag) {

                int ret;

                intra_prediction_unit_default_value(s, x0, y0, log2_cb_size);

                ret = hls_pcm_sample(s, x0, y0, log2_cb_size);

                if (s->sps->pcm.loop_filter_disable_flag)

                    set_deblocking_bypass(s, x0, y0, log2_cb_size);



                if (ret < 0)

                    return ret;

            } else {

                intra_prediction_unit(s, x0, y0, log2_cb_size);

            }

        } else {

            intra_prediction_unit_default_value(s, x0, y0, log2_cb_size);

            switch (lc->cu.part_mode) {

            case PART_2Nx2N:

                hls_prediction_unit(s, x0, y0, cb_size, cb_size, log2_cb_size, 0);

                break;

            case PART_2NxN:

                hls_prediction_unit(s, x0, y0,               cb_size, cb_size / 2, log2_cb_size, 0);

                hls_prediction_unit(s, x0, y0 + cb_size / 2, cb_size, cb_size / 2, log2_cb_size, 1);

                break;

            case PART_Nx2N:

                hls_prediction_unit(s, x0,               y0, cb_size / 2, cb_size, log2_cb_size, 0);

                hls_prediction_unit(s, x0 + cb_size / 2, y0, cb_size / 2, cb_size, log2_cb_size, 1);

                break;

            case PART_2NxnU:

                hls_prediction_unit(s, x0, y0,               cb_size, cb_size     / 4, log2_cb_size, 0);

                hls_prediction_unit(s, x0, y0 + cb_size / 4, cb_size, cb_size * 3 / 4, log2_cb_size, 1);

                break;

            case PART_2NxnD:

                hls_prediction_unit(s, x0, y0,                   cb_size, cb_size * 3 / 4, log2_cb_size, 0);

                hls_prediction_unit(s, x0, y0 + cb_size * 3 / 4, cb_size, cb_size     / 4, log2_cb_size, 1);

                break;

            case PART_nLx2N:

                hls_prediction_unit(s, x0,               y0, cb_size     / 4, cb_size, log2_cb_size, 0);

                hls_prediction_unit(s, x0 + cb_size / 4, y0, cb_size * 3 / 4, cb_size, log2_cb_size, 1);

                break;

            case PART_nRx2N:

                hls_prediction_unit(s, x0,                   y0, cb_size * 3 / 4, cb_size, log2_cb_size, 0);

                hls_prediction_unit(s, x0 + cb_size * 3 / 4, y0, cb_size     / 4, cb_size, log2_cb_size, 1);

                break;

            case PART_NxN:

                hls_prediction_unit(s, x0,               y0,               cb_size / 2, cb_size / 2, log2_cb_size, 0);

                hls_prediction_unit(s, x0 + cb_size / 2, y0,               cb_size / 2, cb_size / 2, log2_cb_size, 1);

                hls_prediction_unit(s, x0,               y0 + cb_size / 2, cb_size / 2, cb_size / 2, log2_cb_size, 2);

                hls_prediction_unit(s, x0 + cb_size / 2, y0 + cb_size / 2, cb_size / 2, cb_size / 2, log2_cb_size, 3);

                break;

            }

        }



        if (!lc->cu.pcm_flag) {

            if (lc->cu.pred_mode != MODE_INTRA &&

                !(lc->cu.part_mode == PART_2Nx2N && lc->pu.merge_flag)) {

                lc->cu.rqt_root_cbf = ff_hevc_no_residual_syntax_flag_decode(s);

            }

            if (lc->cu.rqt_root_cbf) {

                lc->cu.max_trafo_depth = lc->cu.pred_mode == MODE_INTRA ?

                                         s->sps->max_transform_hierarchy_depth_intra + lc->cu.intra_split_flag :

                                         s->sps->max_transform_hierarchy_depth_inter;

                hls_transform_tree(s, x0, y0, x0, y0, x0, y0, log2_cb_size,

                                   log2_cb_size, 0, 0);

            } else {

                if (!s->sh.disable_deblocking_filter_flag)

                    ff_hevc_deblocking_boundary_strengths(s, x0, y0, log2_cb_size,

                                                          lc->slice_or_tiles_up_boundary,

                                                          lc->slice_or_tiles_left_boundary);

            }

        }

    }



    if (s->pps->cu_qp_delta_enabled_flag && lc->tu.is_cu_qp_delta_coded == 0)

        ff_hevc_set_qPy(s, x0, y0, x0, y0, log2_cb_size);



    x = y_cb * min_cb_width + x_cb;

    for (y = 0; y < length; y++) {

        memset(&s->qp_y_tab[x], lc->qp_y, length);

        x += min_cb_width;

    }



    set_ct_depth(s, x0, y0, log2_cb_size, lc->ct.depth);



    return 0;

}
