static int hls_transform_unit(HEVCContext *s, int x0, int y0,

                              int xBase, int yBase, int cb_xBase, int cb_yBase,

                              int log2_cb_size, int log2_trafo_size,

                              int trafo_depth, int blk_idx,

                              int cbf_luma, int *cbf_cb, int *cbf_cr)

{

    HEVCLocalContext *lc = s->HEVClc;

    const int log2_trafo_size_c = log2_trafo_size - s->sps->hshift[1];

    int i;



    if (lc->cu.pred_mode == MODE_INTRA) {

        int trafo_size = 1 << log2_trafo_size;

        ff_hevc_set_neighbour_available(s, x0, y0, trafo_size, trafo_size);



        s->hpc.intra_pred[log2_trafo_size - 2](s, x0, y0, 0);

    }



    if (cbf_luma || cbf_cb[0] || cbf_cr[0] ||

        (s->sps->chroma_format_idc == 2 && (cbf_cb[1] || cbf_cr[1]))) {

        int scan_idx   = SCAN_DIAG;

        int scan_idx_c = SCAN_DIAG;

        int cbf_chroma = cbf_cb[0] || cbf_cr[0] ||

                         (s->sps->chroma_format_idc == 2 &&

                         (cbf_cb[1] || cbf_cr[1]));



        if (s->pps->cu_qp_delta_enabled_flag && !lc->tu.is_cu_qp_delta_coded) {

            lc->tu.cu_qp_delta = ff_hevc_cu_qp_delta_abs(s);

            if (lc->tu.cu_qp_delta != 0)

                if (ff_hevc_cu_qp_delta_sign_flag(s) == 1)

                    lc->tu.cu_qp_delta = -lc->tu.cu_qp_delta;

            lc->tu.is_cu_qp_delta_coded = 1;



            if (lc->tu.cu_qp_delta < -(26 + s->sps->qp_bd_offset / 2) ||

                lc->tu.cu_qp_delta >  (25 + s->sps->qp_bd_offset / 2)) {

                av_log(s->avctx, AV_LOG_ERROR,

                       "The cu_qp_delta %d is outside the valid range "

                       "[%d, %d].\n",

                       lc->tu.cu_qp_delta,

                       -(26 + s->sps->qp_bd_offset / 2),

                        (25 + s->sps->qp_bd_offset / 2));

                return AVERROR_INVALIDDATA;

            }



            ff_hevc_set_qPy(s, cb_xBase, cb_yBase, log2_cb_size);

        }



        if (s->sh.cu_chroma_qp_offset_enabled_flag && cbf_chroma &&

            !lc->cu.cu_transquant_bypass_flag  &&  !lc->tu.is_cu_chroma_qp_offset_coded) {

            int cu_chroma_qp_offset_flag = ff_hevc_cu_chroma_qp_offset_flag(s);

            if (cu_chroma_qp_offset_flag) {

                int cu_chroma_qp_offset_idx  = 0;

                if (s->pps->chroma_qp_offset_list_len_minus1 > 0) {

                    cu_chroma_qp_offset_idx = ff_hevc_cu_chroma_qp_offset_idx(s);

                    av_log(s->avctx, AV_LOG_ERROR,

                        "cu_chroma_qp_offset_idx not yet tested.\n");

                }

                lc->tu.cu_qp_offset_cb = s->pps->cb_qp_offset_list[cu_chroma_qp_offset_idx];

                lc->tu.cu_qp_offset_cr = s->pps->cr_qp_offset_list[cu_chroma_qp_offset_idx];

            } else {

                lc->tu.cu_qp_offset_cb = 0;

                lc->tu.cu_qp_offset_cr = 0;

            }

            lc->tu.is_cu_chroma_qp_offset_coded = 1;

        }



        if (lc->cu.pred_mode == MODE_INTRA && log2_trafo_size < 4) {

            if (lc->tu.intra_pred_mode >= 6 &&

                lc->tu.intra_pred_mode <= 14) {

                scan_idx = SCAN_VERT;

            } else if (lc->tu.intra_pred_mode >= 22 &&

                       lc->tu.intra_pred_mode <= 30) {

                scan_idx = SCAN_HORIZ;

            }



            if (lc->tu.intra_pred_mode_c >=  6 &&

                lc->tu.intra_pred_mode_c <= 14) {

                scan_idx_c = SCAN_VERT;

            } else if (lc->tu.intra_pred_mode_c >= 22 &&

                       lc->tu.intra_pred_mode_c <= 30) {

                scan_idx_c = SCAN_HORIZ;

            }

        }



        lc->tu.cross_pf = 0;



        if (cbf_luma)

            ff_hevc_hls_residual_coding(s, x0, y0, log2_trafo_size, scan_idx, 0);

        if (log2_trafo_size > 2 || s->sps->chroma_format_idc == 3) {

            int trafo_size_h = 1 << (log2_trafo_size_c + s->sps->hshift[1]);

            int trafo_size_v = 1 << (log2_trafo_size_c + s->sps->vshift[1]);

            lc->tu.cross_pf  = (s->pps->cross_component_prediction_enabled_flag && cbf_luma &&

                                (lc->cu.pred_mode == MODE_INTER ||

                                 (lc->tu.chroma_mode_c ==  4)));



            if (lc->tu.cross_pf) {

                hls_cross_component_pred(s, 0);

            }

            for (i = 0; i < (s->sps->chroma_format_idc == 2 ? 2 : 1); i++) {

                if (lc->cu.pred_mode == MODE_INTRA) {

                    ff_hevc_set_neighbour_available(s, x0, y0 + (i << log2_trafo_size_c), trafo_size_h, trafo_size_v);

                    s->hpc.intra_pred[log2_trafo_size_c - 2](s, x0, y0 + (i << log2_trafo_size_c), 1);

                }

                if (cbf_cb[i])

                    ff_hevc_hls_residual_coding(s, x0, y0 + (i << log2_trafo_size_c),

                                                log2_trafo_size_c, scan_idx_c, 1);

                else

                    if (lc->tu.cross_pf) {

                        ptrdiff_t stride = s->frame->linesize[1];

                        int hshift = s->sps->hshift[1];

                        int vshift = s->sps->vshift[1];

                        int16_t *coeffs_y = lc->tu.coeffs[0];

                        int16_t *coeffs =   lc->tu.coeffs[1];

                        int size = 1 << log2_trafo_size_c;



                        uint8_t *dst = &s->frame->data[1][(y0 >> vshift) * stride +

                                                              ((x0 >> hshift) << s->sps->pixel_shift)];

                        for (i = 0; i < (size * size); i++) {

                            coeffs[i] = ((lc->tu.res_scale_val * coeffs_y[i]) >> 3);

                        }

                        s->hevcdsp.transform_add[log2_trafo_size-2](dst, coeffs, stride);

                    }

            }



            if (lc->tu.cross_pf) {

                hls_cross_component_pred(s, 1);

            }

            for (i = 0; i < (s->sps->chroma_format_idc == 2 ? 2 : 1); i++) {

                if (lc->cu.pred_mode == MODE_INTRA) {

                    ff_hevc_set_neighbour_available(s, x0, y0 + (i << log2_trafo_size_c), trafo_size_h, trafo_size_v);

                    s->hpc.intra_pred[log2_trafo_size_c - 2](s, x0, y0 + (i << log2_trafo_size_c), 2);

                }

                if (cbf_cr[i])

                    ff_hevc_hls_residual_coding(s, x0, y0 + (i << log2_trafo_size_c),

                                                log2_trafo_size_c, scan_idx_c, 2);

                else

                    if (lc->tu.cross_pf) {

                        ptrdiff_t stride = s->frame->linesize[2];

                        int hshift = s->sps->hshift[2];

                        int vshift = s->sps->vshift[2];

                        int16_t *coeffs_y = lc->tu.coeffs[0];

                        int16_t *coeffs =   lc->tu.coeffs[1];

                        int size = 1 << log2_trafo_size_c;



                        uint8_t *dst = &s->frame->data[2][(y0 >> vshift) * stride +

                                                          ((x0 >> hshift) << s->sps->pixel_shift)];

                        for (i = 0; i < (size * size); i++) {

                            coeffs[i] = ((lc->tu.res_scale_val * coeffs_y[i]) >> 3);

                        }

                        s->hevcdsp.transform_add[log2_trafo_size-2](dst, coeffs, stride);

                    }

            }

        } else if (blk_idx == 3) {

            int trafo_size_h = 1 << (log2_trafo_size + 1);

            int trafo_size_v = 1 << (log2_trafo_size + s->sps->vshift[1]);

            for (i = 0; i < (s->sps->chroma_format_idc == 2 ? 2 : 1); i++) {

                if (lc->cu.pred_mode == MODE_INTRA) {

                    ff_hevc_set_neighbour_available(s, xBase, yBase + (i << log2_trafo_size),

                                                    trafo_size_h, trafo_size_v);

                    s->hpc.intra_pred[log2_trafo_size - 2](s, xBase, yBase + (i << log2_trafo_size), 1);

                }

                if (cbf_cb[i])

                    ff_hevc_hls_residual_coding(s, xBase, yBase + (i << log2_trafo_size),

                                                log2_trafo_size, scan_idx_c, 1);

            }

            for (i = 0; i < (s->sps->chroma_format_idc == 2 ? 2 : 1); i++) {

                if (lc->cu.pred_mode == MODE_INTRA) {

                    ff_hevc_set_neighbour_available(s, xBase, yBase + (i << log2_trafo_size),

                                                trafo_size_h, trafo_size_v);

                    s->hpc.intra_pred[log2_trafo_size - 2](s, xBase, yBase + (i << log2_trafo_size), 2);

                }

                if (cbf_cr[i])

                    ff_hevc_hls_residual_coding(s, xBase, yBase + (i << log2_trafo_size),

                                                log2_trafo_size, scan_idx_c, 2);

            }

        }

    } else if (lc->cu.pred_mode == MODE_INTRA) {

        if (log2_trafo_size > 2 || s->sps->chroma_format_idc == 3) {

            int trafo_size_h = 1 << (log2_trafo_size_c + s->sps->hshift[1]);

            int trafo_size_v = 1 << (log2_trafo_size_c + s->sps->vshift[1]);

            ff_hevc_set_neighbour_available(s, x0, y0, trafo_size_h, trafo_size_v);

            s->hpc.intra_pred[log2_trafo_size_c - 2](s, x0, y0, 1);

            s->hpc.intra_pred[log2_trafo_size_c - 2](s, x0, y0, 2);

            if (s->sps->chroma_format_idc == 2) {

                ff_hevc_set_neighbour_available(s, x0, y0 + (1 << log2_trafo_size_c),

                                                trafo_size_h, trafo_size_v);

                s->hpc.intra_pred[log2_trafo_size_c - 2](s, x0, y0 + (1 << log2_trafo_size_c), 1);

                s->hpc.intra_pred[log2_trafo_size_c - 2](s, x0, y0 + (1 << log2_trafo_size_c), 2);

            }

        } else if (blk_idx == 3) {

            int trafo_size_h = 1 << (log2_trafo_size + 1);

            int trafo_size_v = 1 << (log2_trafo_size + s->sps->vshift[1]);

            ff_hevc_set_neighbour_available(s, xBase, yBase,

                                            trafo_size_h, trafo_size_v);

            s->hpc.intra_pred[log2_trafo_size - 2](s, xBase, yBase, 1);

            s->hpc.intra_pred[log2_trafo_size - 2](s, xBase, yBase, 2);

            if (s->sps->chroma_format_idc == 2) {

                ff_hevc_set_neighbour_available(s, xBase, yBase + (1 << (log2_trafo_size)),

                                                trafo_size_h, trafo_size_v);

                s->hpc.intra_pred[log2_trafo_size - 2](s, xBase, yBase + (1 << (log2_trafo_size)), 1);

                s->hpc.intra_pred[log2_trafo_size - 2](s, xBase, yBase + (1 << (log2_trafo_size)), 2);

            }

        }

    }



    return 0;

}
