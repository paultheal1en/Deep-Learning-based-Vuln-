void ff_hevc_deblocking_boundary_strengths(HEVCContext *s, int x0, int y0,

                                           int log2_trafo_size)

{

    HEVCLocalContext *lc = &s->HEVClc;

    MvField *tab_mvf     = s->ref->tab_mvf;

    int log2_min_pu_size = s->sps->log2_min_pu_size;

    int log2_min_tu_size = s->sps->log2_min_tb_size;

    int min_pu_width     = s->sps->min_pu_width;

    int min_tu_width     = s->sps->min_tb_width;

    int is_intra = tab_mvf[(y0 >> log2_min_pu_size) * min_pu_width +

                           (x0 >> log2_min_pu_size)].is_intra;

    int i, j, bs;



    if (y0 > 0 && (y0 & 7) == 0) {

        int yp_pu = (y0 - 1) >> log2_min_pu_size;

        int yq_pu =  y0      >> log2_min_pu_size;

        int yp_tu = (y0 - 1) >> log2_min_tu_size;

        int yq_tu =  y0      >> log2_min_tu_size;



        for (i = 0; i < (1 << log2_trafo_size); i += 4) {

            int x_pu = (x0 + i) >> log2_min_pu_size;

            int x_tu = (x0 + i) >> log2_min_tu_size;

            MvField *top  = &tab_mvf[yp_pu * min_pu_width + x_pu];

            MvField *curr = &tab_mvf[yq_pu * min_pu_width + x_pu];

            uint8_t top_cbf_luma  = s->cbf_luma[yp_tu * min_tu_width + x_tu];

            uint8_t curr_cbf_luma = s->cbf_luma[yq_tu * min_tu_width + x_tu];

            RefPicList *top_refPicList = ff_hevc_get_ref_list(s, s->ref,

                                                              x0 + i, y0 - 1);



            bs = boundary_strength(s, curr, curr_cbf_luma,

                                   top, top_cbf_luma, top_refPicList, 1);

            if (!s->sh.slice_loop_filter_across_slices_enabled_flag &&

                lc->boundary_flags & BOUNDARY_UPPER_SLICE &&

                (y0 % (1 << s->sps->log2_ctb_size)) == 0)

                bs = 0;

            else if (!s->pps->loop_filter_across_tiles_enabled_flag &&

                     lc->boundary_flags & BOUNDARY_UPPER_TILE &&

                     (y0 % (1 << s->sps->log2_ctb_size)) == 0)

                bs = 0;

            if (bs)

                s->horizontal_bs[((x0 + i) + y0 * s->bs_width) >> 2] = bs;

        }

    }



    // bs for TU internal horizontal PU boundaries

    if (log2_trafo_size > s->sps->log2_min_pu_size && !is_intra)

        for (j = 8; j < (1 << log2_trafo_size); j += 8) {

            int yp_pu = (y0 + j - 1) >> log2_min_pu_size;

            int yq_pu = (y0 + j)     >> log2_min_pu_size;

            int yp_tu = (y0 + j - 1) >> log2_min_tu_size;

            int yq_tu = (y0 + j)     >> log2_min_tu_size;



            for (i = 0; i < (1 << log2_trafo_size); i += 4) {

                int x_pu = (x0 + i) >> log2_min_pu_size;

                int x_tu = (x0 + i) >> log2_min_tu_size;

                MvField *top  = &tab_mvf[yp_pu * min_pu_width + x_pu];

                MvField *curr = &tab_mvf[yq_pu * min_pu_width + x_pu];

                uint8_t top_cbf_luma  = s->cbf_luma[yp_tu * min_tu_width + x_tu];

                uint8_t curr_cbf_luma = s->cbf_luma[yq_tu * min_tu_width + x_tu];

                RefPicList *top_refPicList = ff_hevc_get_ref_list(s, s->ref,

                                                                  x0 + i,

                                                                  y0 + j - 1);



                bs = boundary_strength(s, curr, curr_cbf_luma,

                                       top, top_cbf_luma, top_refPicList, 0);

                if (bs)

                    s->horizontal_bs[((x0 + i) + (y0 + j) * s->bs_width) >> 2] = bs;

            }

        }



    // bs for vertical TU boundaries

    if (x0 > 0 && (x0 & 7) == 0) {

        int xp_pu = (x0 - 1) >> log2_min_pu_size;

        int xq_pu =  x0      >> log2_min_pu_size;

        int xp_tu = (x0 - 1) >> log2_min_tu_size;

        int xq_tu =  x0      >> log2_min_tu_size;



        for (i = 0; i < (1 << log2_trafo_size); i += 4) {

            int y_pu      = (y0 + i) >> log2_min_pu_size;

            int y_tu      = (y0 + i) >> log2_min_tu_size;

            MvField *left = &tab_mvf[y_pu * min_pu_width + xp_pu];

            MvField *curr = &tab_mvf[y_pu * min_pu_width + xq_pu];



            uint8_t left_cbf_luma = s->cbf_luma[y_tu * min_tu_width + xp_tu];

            uint8_t curr_cbf_luma = s->cbf_luma[y_tu * min_tu_width + xq_tu];

            RefPicList *left_refPicList = ff_hevc_get_ref_list(s, s->ref,

                                                               x0 - 1, y0 + i);



            bs = boundary_strength(s, curr, curr_cbf_luma,

                                   left, left_cbf_luma, left_refPicList, 1);

            if (!s->sh.slice_loop_filter_across_slices_enabled_flag &&

                lc->boundary_flags & BOUNDARY_LEFT_SLICE &&

                (x0 % (1 << s->sps->log2_ctb_size)) == 0)

                bs = 0;

            else if (!s->pps->loop_filter_across_tiles_enabled_flag &&

                     lc->boundary_flags & BOUNDARY_LEFT_TILE &&

                     (x0 % (1 << s->sps->log2_ctb_size)) == 0)

                bs = 0;

            if (bs)

                s->vertical_bs[(x0 >> 3) + ((y0 + i) >> 2) * s->bs_width] = bs;

        }

    }



    // bs for TU internal vertical PU boundaries

    if (log2_trafo_size > log2_min_pu_size && !is_intra)

        for (j = 0; j < (1 << log2_trafo_size); j += 4) {

            int y_pu = (y0 + j) >> log2_min_pu_size;

            int y_tu = (y0 + j) >> log2_min_tu_size;



            for (i = 8; i < (1 << log2_trafo_size); i += 8) {

                int xp_pu = (x0 + i - 1) >> log2_min_pu_size;

                int xq_pu = (x0 + i)     >> log2_min_pu_size;

                int xp_tu = (x0 + i - 1) >> log2_min_tu_size;

                int xq_tu = (x0 + i)     >> log2_min_tu_size;

                MvField *left = &tab_mvf[y_pu * min_pu_width + xp_pu];

                MvField *curr = &tab_mvf[y_pu * min_pu_width + xq_pu];

                uint8_t left_cbf_luma = s->cbf_luma[y_tu * min_tu_width + xp_tu];

                uint8_t curr_cbf_luma = s->cbf_luma[y_tu * min_tu_width + xq_tu];

                RefPicList *left_refPicList = ff_hevc_get_ref_list(s, s->ref,

                                                                   x0 + i - 1,

                                                                   y0 + j);



                bs = boundary_strength(s, curr, curr_cbf_luma,

                                       left, left_cbf_luma, left_refPicList, 0);

                if (bs)

                    s->vertical_bs[((x0 + i) >> 3) + ((y0 + j) >> 2) * s->bs_width] = bs;

            }

        }

}
