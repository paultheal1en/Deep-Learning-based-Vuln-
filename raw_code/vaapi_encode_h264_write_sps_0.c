static void vaapi_encode_h264_write_sps(PutBitContext *pbc,

                                        VAAPIEncodeContext *ctx)

{

    VAEncSequenceParameterBufferH264  *vseq = ctx->codec_sequence_params;

    VAAPIEncodeH264Context            *priv = ctx->priv_data;

    VAAPIEncodeH264MiscSequenceParams *mseq = &priv->misc_sequence_params;

    int i;



    vaapi_encode_h264_write_nal_header(pbc, NAL_SPS, 3);



    u(8, mseq_var(profile_idc));

    u(1, mseq_var(constraint_set0_flag));

    u(1, mseq_var(constraint_set1_flag));

    u(1, mseq_var(constraint_set2_flag));

    u(1, mseq_var(constraint_set3_flag));

    u(1, mseq_var(constraint_set4_flag));

    u(1, mseq_var(constraint_set5_flag));

    u(2, 0, reserved_zero_2bits);



    u(8, vseq_var(level_idc));



    ue(vseq_var(seq_parameter_set_id));



    if (mseq->profile_idc == 100 || mseq->profile_idc == 110 ||

        mseq->profile_idc == 122 || mseq->profile_idc == 244 ||

        mseq->profile_idc ==  44 || mseq->profile_idc ==  83 ||

        mseq->profile_idc ==  86 || mseq->profile_idc == 118 ||

        mseq->profile_idc == 128 || mseq->profile_idc == 138) {

        ue(vseq_field(chroma_format_idc));



        if (vseq->seq_fields.bits.chroma_format_idc == 3)

            u(1, mseq_var(separate_colour_plane_flag));



        ue(vseq_var(bit_depth_luma_minus8));

        ue(vseq_var(bit_depth_chroma_minus8));



        u(1, mseq_var(qpprime_y_zero_transform_bypass_flag));



        u(1, vseq_field(seq_scaling_matrix_present_flag));

        if (vseq->seq_fields.bits.seq_scaling_matrix_present_flag) {

            av_assert0(0 && "scaling matrices not supported");

        }

    }



    ue(vseq_field(log2_max_frame_num_minus4));

    ue(vseq_field(pic_order_cnt_type));



    if (vseq->seq_fields.bits.pic_order_cnt_type == 0) {

        ue(vseq_field(log2_max_pic_order_cnt_lsb_minus4));

    } else if (vseq->seq_fields.bits.pic_order_cnt_type == 1) {

        u(1, mseq_var(delta_pic_order_always_zero_flag));

        se(vseq_var(offset_for_non_ref_pic));

        se(vseq_var(offset_for_top_to_bottom_field));

        ue(vseq_var(num_ref_frames_in_pic_order_cnt_cycle));



        for (i = 0; i < vseq->num_ref_frames_in_pic_order_cnt_cycle; i++)

            se(vseq_var(offset_for_ref_frame[i]));

    }



    ue(vseq_var(max_num_ref_frames));

    u(1, mseq_var(gaps_in_frame_num_allowed_flag));



    ue(vseq->picture_width_in_mbs  - 1, pic_width_in_mbs_minus1);

    ue(vseq->picture_height_in_mbs - 1, pic_height_in_mbs_minus1);



    u(1, vseq_field(frame_mbs_only_flag));

    if (!vseq->seq_fields.bits.frame_mbs_only_flag)

        u(1, vseq_field(mb_adaptive_frame_field_flag));



    u(1, vseq_field(direct_8x8_inference_flag));



    u(1, vseq_var(frame_cropping_flag));

    if (vseq->frame_cropping_flag) {

        ue(vseq_var(frame_crop_left_offset));

        ue(vseq_var(frame_crop_right_offset));

        ue(vseq_var(frame_crop_top_offset));

        ue(vseq_var(frame_crop_bottom_offset));

    }



    u(1, vseq_var(vui_parameters_present_flag));

    if (vseq->vui_parameters_present_flag)

        vaapi_encode_h264_write_vui(pbc, ctx);



    vaapi_encode_h264_write_trailing_rbsp(pbc);

}
