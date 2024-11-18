static int vaapi_h264_start_frame(AVCodecContext          *avctx,

                                  av_unused const uint8_t *buffer,

                                  av_unused uint32_t       size)

{

    H264Context * const h = avctx->priv_data;

    struct vaapi_context * const vactx = avctx->hwaccel_context;

    VAPictureParameterBufferH264 *pic_param;

    VAIQMatrixBufferH264 *iq_matrix;



    ff_dlog(avctx, "vaapi_h264_start_frame()\n");



    vactx->slice_param_size = sizeof(VASliceParameterBufferH264);



    /* Fill in VAPictureParameterBufferH264. */

    pic_param = ff_vaapi_alloc_pic_param(vactx, sizeof(VAPictureParameterBufferH264));

    if (!pic_param)

        return -1;

    fill_vaapi_pic(&pic_param->CurrPic, h->cur_pic_ptr, h->picture_structure);

    if (fill_vaapi_ReferenceFrames(pic_param, h) < 0)

        return -1;

    pic_param->picture_width_in_mbs_minus1                      = h->mb_width - 1;

    pic_param->picture_height_in_mbs_minus1                     = h->mb_height - 1;

    pic_param->bit_depth_luma_minus8                            = h->sps.bit_depth_luma - 8;

    pic_param->bit_depth_chroma_minus8                          = h->sps.bit_depth_chroma - 8;

    pic_param->num_ref_frames                                   = h->sps.ref_frame_count;

    pic_param->seq_fields.value                                 = 0; /* reset all bits */

    pic_param->seq_fields.bits.chroma_format_idc                = h->sps.chroma_format_idc;

    pic_param->seq_fields.bits.residual_colour_transform_flag   = h->sps.residual_color_transform_flag; /* XXX: only for 4:4:4 high profile? */

    pic_param->seq_fields.bits.gaps_in_frame_num_value_allowed_flag = h->sps.gaps_in_frame_num_allowed_flag;

    pic_param->seq_fields.bits.frame_mbs_only_flag              = h->sps.frame_mbs_only_flag;

    pic_param->seq_fields.bits.mb_adaptive_frame_field_flag     = h->sps.mb_aff;

    pic_param->seq_fields.bits.direct_8x8_inference_flag        = h->sps.direct_8x8_inference_flag;

    pic_param->seq_fields.bits.MinLumaBiPredSize8x8             = h->sps.level_idc >= 31; /* A.3.3.2 */

    pic_param->seq_fields.bits.log2_max_frame_num_minus4        = h->sps.log2_max_frame_num - 4;

    pic_param->seq_fields.bits.pic_order_cnt_type               = h->sps.poc_type;

    pic_param->seq_fields.bits.log2_max_pic_order_cnt_lsb_minus4 = h->sps.log2_max_poc_lsb - 4;

    pic_param->seq_fields.bits.delta_pic_order_always_zero_flag = h->sps.delta_pic_order_always_zero_flag;

    pic_param->num_slice_groups_minus1                          = h->pps.slice_group_count - 1;

    pic_param->slice_group_map_type                             = h->pps.mb_slice_group_map_type;

    pic_param->slice_group_change_rate_minus1                   = 0; /* XXX: unimplemented in Libav */

    pic_param->pic_init_qp_minus26                              = h->pps.init_qp - 26;

    pic_param->pic_init_qs_minus26                              = h->pps.init_qs - 26;

    pic_param->chroma_qp_index_offset                           = h->pps.chroma_qp_index_offset[0];

    pic_param->second_chroma_qp_index_offset                    = h->pps.chroma_qp_index_offset[1];

    pic_param->pic_fields.value                                 = 0; /* reset all bits */

    pic_param->pic_fields.bits.entropy_coding_mode_flag         = h->pps.cabac;

    pic_param->pic_fields.bits.weighted_pred_flag               = h->pps.weighted_pred;

    pic_param->pic_fields.bits.weighted_bipred_idc              = h->pps.weighted_bipred_idc;

    pic_param->pic_fields.bits.transform_8x8_mode_flag          = h->pps.transform_8x8_mode;

    pic_param->pic_fields.bits.field_pic_flag                   = h->picture_structure != PICT_FRAME;

    pic_param->pic_fields.bits.constrained_intra_pred_flag      = h->pps.constrained_intra_pred;

    pic_param->pic_fields.bits.pic_order_present_flag           = h->pps.pic_order_present;

    pic_param->pic_fields.bits.deblocking_filter_control_present_flag = h->pps.deblocking_filter_parameters_present;

    pic_param->pic_fields.bits.redundant_pic_cnt_present_flag   = h->pps.redundant_pic_cnt_present;

    pic_param->pic_fields.bits.reference_pic_flag               = h->nal_ref_idc != 0;

    pic_param->frame_num                                        = h->frame_num;



    /* Fill in VAIQMatrixBufferH264. */

    iq_matrix = ff_vaapi_alloc_iq_matrix(vactx, sizeof(VAIQMatrixBufferH264));

    if (!iq_matrix)

        return -1;

    memcpy(iq_matrix->ScalingList4x4, h->pps.scaling_matrix4, sizeof(iq_matrix->ScalingList4x4));

    memcpy(iq_matrix->ScalingList8x8[0], h->pps.scaling_matrix8[0], sizeof(iq_matrix->ScalingList8x8[0]));

    memcpy(iq_matrix->ScalingList8x8[1], h->pps.scaling_matrix8[3], sizeof(iq_matrix->ScalingList8x8[0]));

    return 0;

}
