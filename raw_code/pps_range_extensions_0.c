static int pps_range_extensions(GetBitContext *gb, AVCodecContext *avctx,

                                HEVCPPS *pps, HEVCSPS *sps) {

    int i;



    if (pps->transform_skip_enabled_flag) {

        pps->log2_max_transform_skip_block_size = get_ue_golomb_long(gb) + 2;

    }

    pps->cross_component_prediction_enabled_flag = get_bits1(gb);

    pps->chroma_qp_offset_list_enabled_flag = get_bits1(gb);

    if (pps->chroma_qp_offset_list_enabled_flag) {

        pps->diff_cu_chroma_qp_offset_depth = get_ue_golomb_long(gb);

        pps->chroma_qp_offset_list_len_minus1 = get_ue_golomb_long(gb);

        if (pps->chroma_qp_offset_list_len_minus1 && pps->chroma_qp_offset_list_len_minus1 >= 5) {

            av_log(avctx, AV_LOG_ERROR,

                   "chroma_qp_offset_list_len_minus1 shall be in the range [0, 5].\n");

            return AVERROR_INVALIDDATA;

        }

        for (i = 0; i <= pps->chroma_qp_offset_list_len_minus1; i++) {

            pps->cb_qp_offset_list[i] = get_se_golomb_long(gb);

            if (pps->cb_qp_offset_list[i]) {

                av_log(avctx, AV_LOG_WARNING,

                       "cb_qp_offset_list not tested yet.\n");

            }

            pps->cr_qp_offset_list[i] = get_se_golomb_long(gb);

            if (pps->cr_qp_offset_list[i]) {

                av_log(avctx, AV_LOG_WARNING,

                       "cb_qp_offset_list not tested yet.\n");

            }

        }

    }

    pps->log2_sao_offset_scale_luma = get_ue_golomb_long(gb);

    pps->log2_sao_offset_scale_chroma = get_ue_golomb_long(gb);



    return(0);

}
