static int decode_slice(AVCodecContext *avctx,

                        const uint8_t  *buffer,

                        uint32_t        size)

{

    H264Context * const h = avctx->priv_data;

    MpegEncContext * const s = &h->s;

    VASliceParameterBufferH264 *slice_param;



    dprintf(avctx, "decode_slice(): buffer %p, size %d\n", buffer, size);



    /* Fill in VASliceParameterBufferH264. */

    slice_param = (VASliceParameterBufferH264 *)ff_vaapi_alloc_slice(avctx->hwaccel_context, buffer, size);

    if (!slice_param)

        return -1;

    slice_param->slice_data_bit_offset          = get_bits_count(&h->s.gb) + 8; /* bit buffer started beyond nal_unit_type */

    slice_param->first_mb_in_slice              = (s->mb_y >> FIELD_OR_MBAFF_PICTURE) * s->mb_width + s->mb_x;

    slice_param->slice_type                     = ff_h264_get_slice_type(h);

    slice_param->direct_spatial_mv_pred_flag    = h->slice_type == FF_B_TYPE ? h->direct_spatial_mv_pred : 0;

    slice_param->num_ref_idx_l0_active_minus1   = h->list_count > 0 ? h->ref_count[0] - 1 : 0;

    slice_param->num_ref_idx_l1_active_minus1   = h->list_count > 1 ? h->ref_count[1] - 1 : 0;

    slice_param->cabac_init_idc                 = h->cabac_init_idc;

    slice_param->slice_qp_delta                 = s->qscale - h->pps.init_qp;

    slice_param->disable_deblocking_filter_idc  = h->deblocking_filter < 2 ? !h->deblocking_filter : h->deblocking_filter;

    slice_param->slice_alpha_c0_offset_div2     = h->slice_alpha_c0_offset / 2;

    slice_param->slice_beta_offset_div2         = h->slice_beta_offset / 2;

    slice_param->luma_log2_weight_denom         = h->luma_log2_weight_denom;

    slice_param->chroma_log2_weight_denom       = h->chroma_log2_weight_denom;



    fill_vaapi_RefPicList(slice_param->RefPicList0, h->ref_list[0], h->list_count > 0 ? h->ref_count[0] : 0);

    fill_vaapi_RefPicList(slice_param->RefPicList1, h->ref_list[1], h->list_count > 1 ? h->ref_count[1] : 0);



    fill_vaapi_plain_pred_weight_table(h, 0,

                                       &slice_param->luma_weight_l0_flag,   slice_param->luma_weight_l0,   slice_param->luma_offset_l0,

                                       &slice_param->chroma_weight_l0_flag, slice_param->chroma_weight_l0, slice_param->chroma_offset_l0);

    fill_vaapi_plain_pred_weight_table(h, 1,

                                       &slice_param->luma_weight_l1_flag,   slice_param->luma_weight_l1,   slice_param->luma_offset_l1,

                                       &slice_param->chroma_weight_l1_flag, slice_param->chroma_weight_l1, slice_param->chroma_offset_l1);

    return 0;

}
