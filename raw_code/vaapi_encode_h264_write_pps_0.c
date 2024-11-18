static void vaapi_encode_h264_write_pps(PutBitContext *pbc,

                                        VAAPIEncodeContext *ctx)

{

    VAEncPictureParameterBufferH264   *vpic = ctx->codec_picture_params;

    VAAPIEncodeH264Context            *priv = ctx->priv_data;

    VAAPIEncodeH264MiscSequenceParams *mseq = &priv->misc_sequence_params;



    vaapi_encode_h264_write_nal_header(pbc, NAL_PPS, 3);



    ue(vpic_var(pic_parameter_set_id));

    ue(vpic_var(seq_parameter_set_id));



    u(1, vpic_field(entropy_coding_mode_flag));

    u(1, mseq_var(bottom_field_pic_order_in_frame_present_flag));



    ue(mseq_var(num_slice_groups_minus1));

    if (mseq->num_slice_groups_minus1 > 0) {

        ue(mseq_var(slice_group_map_type));

        av_assert0(0 && "slice groups not supported");

    }



    ue(vpic_var(num_ref_idx_l0_active_minus1));

    ue(vpic_var(num_ref_idx_l1_active_minus1));



    u(1, vpic_field(weighted_pred_flag));

    u(2, vpic_field(weighted_bipred_idc));



    se(vpic->pic_init_qp - 26, pic_init_qp_minus26);

    se(mseq_var(pic_init_qs_minus26));

    se(vpic_var(chroma_qp_index_offset));



    u(1, vpic_field(deblocking_filter_control_present_flag));

    u(1, vpic_field(constrained_intra_pred_flag));

    u(1, vpic_field(redundant_pic_cnt_present_flag));

    u(1, vpic_field(transform_8x8_mode_flag));



    u(1, vpic_field(pic_scaling_matrix_present_flag));

    if (vpic->pic_fields.bits.pic_scaling_matrix_present_flag) {

        av_assert0(0 && "scaling matrices not supported");

    }



    se(vpic_var(second_chroma_qp_index_offset));



    vaapi_encode_h264_write_trailing_rbsp(pbc);

}
