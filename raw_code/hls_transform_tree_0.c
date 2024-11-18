static void hls_transform_tree(HEVCContext *s, int x0, int y0,

                               int xBase, int yBase, int cb_xBase, int cb_yBase,

                               int log2_cb_size, int log2_trafo_size,

                               int trafo_depth, int blk_idx)

{

    HEVCLocalContext *lc = &s->HEVClc;

    uint8_t split_transform_flag;



    if (trafo_depth > 0 && log2_trafo_size == 2) {

        SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth], x0, y0) =

            SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth - 1], xBase, yBase);

        SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth], x0, y0) =

            SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth - 1], xBase, yBase);

    } else {

        SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth], x0, y0) =

        SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth], x0, y0) = 0;

    }



    if (lc->cu.intra_split_flag) {

        if (trafo_depth == 1)

            lc->tu.cur_intra_pred_mode = lc->pu.intra_pred_mode[blk_idx];

    } else {

        lc->tu.cur_intra_pred_mode = lc->pu.intra_pred_mode[0];

    }



    lc->tt.cbf_luma = 1;



    lc->tt.inter_split_flag = s->sps->max_transform_hierarchy_depth_inter == 0 &&

                              lc->cu.pred_mode == MODE_INTER &&

                              lc->cu.part_mode != PART_2Nx2N &&

                              trafo_depth == 0;



    if (log2_trafo_size <= s->sps->log2_max_trafo_size &&

        log2_trafo_size >  s->sps->log2_min_tb_size    &&

        trafo_depth     < lc->cu.max_trafo_depth       &&

        !(lc->cu.intra_split_flag && trafo_depth == 0)) {

        split_transform_flag = ff_hevc_split_transform_flag_decode(s, log2_trafo_size);

    } else {

        split_transform_flag = log2_trafo_size > s->sps->log2_max_trafo_size ||

                               (lc->cu.intra_split_flag && trafo_depth == 0) ||

                               lc->tt.inter_split_flag;

    }



    if (log2_trafo_size > 2) {

        if (trafo_depth == 0 ||

            SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth - 1], xBase, yBase)) {

            SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth], x0, y0) =

                ff_hevc_cbf_cb_cr_decode(s, trafo_depth);

        }



        if (trafo_depth == 0 ||

            SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth - 1], xBase, yBase)) {

            SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth], x0, y0) =

                ff_hevc_cbf_cb_cr_decode(s, trafo_depth);

        }

    }



    if (split_transform_flag) {

        int x1 = x0 + ((1 << log2_trafo_size) >> 1);

        int y1 = y0 + ((1 << log2_trafo_size) >> 1);



        hls_transform_tree(s, x0, y0, x0, y0, cb_xBase, cb_yBase, log2_cb_size,

                           log2_trafo_size - 1, trafo_depth + 1, 0);

        hls_transform_tree(s, x1, y0, x0, y0, cb_xBase, cb_yBase, log2_cb_size,

                           log2_trafo_size - 1, trafo_depth + 1, 1);

        hls_transform_tree(s, x0, y1, x0, y0, cb_xBase, cb_yBase, log2_cb_size,

                           log2_trafo_size - 1, trafo_depth + 1, 2);

        hls_transform_tree(s, x1, y1, x0, y0, cb_xBase, cb_yBase, log2_cb_size,

                           log2_trafo_size - 1, trafo_depth + 1, 3);

    } else {

        int min_tu_size      = 1 << s->sps->log2_min_tb_size;

        int log2_min_tu_size = s->sps->log2_min_tb_size;

        int min_tu_width     = s->sps->min_tb_width;



        if (lc->cu.pred_mode == MODE_INTRA || trafo_depth != 0 ||

            SAMPLE_CBF(lc->tt.cbf_cb[trafo_depth], x0, y0) ||

            SAMPLE_CBF(lc->tt.cbf_cr[trafo_depth], x0, y0)) {

            lc->tt.cbf_luma = ff_hevc_cbf_luma_decode(s, trafo_depth);

        }



        hls_transform_unit(s, x0, y0, xBase, yBase, cb_xBase, cb_yBase,

                           log2_cb_size, log2_trafo_size, trafo_depth, blk_idx);



        // TODO: store cbf_luma somewhere else

        if (lc->tt.cbf_luma) {

            int i, j;

            for (i = 0; i < (1 << log2_trafo_size); i += min_tu_size)

                for (j = 0; j < (1 << log2_trafo_size); j += min_tu_size) {

                    int x_tu = (x0 + j) >> log2_min_tu_size;

                    int y_tu = (y0 + i) >> log2_min_tu_size;

                    s->cbf_luma[y_tu * min_tu_width + x_tu] = 1;

                }

        }

        if (!s->sh.disable_deblocking_filter_flag) {

            ff_hevc_deblocking_boundary_strengths(s, x0, y0, log2_trafo_size,

                                                  lc->slice_or_tiles_up_boundary,

                                                  lc->slice_or_tiles_left_boundary);

            if (s->pps->transquant_bypass_enable_flag &&

                lc->cu.cu_transquant_bypass_flag)

                set_deblocking_bypass(s, x0, y0, log2_trafo_size);

        }

    }

}
