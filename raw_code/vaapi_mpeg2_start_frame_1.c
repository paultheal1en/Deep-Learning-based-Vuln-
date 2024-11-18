static int vaapi_mpeg2_start_frame(AVCodecContext *avctx, av_unused const uint8_t *buffer, av_unused uint32_t size)

{

    struct MpegEncContext * const s = avctx->priv_data;

    struct vaapi_context * const vactx = avctx->hwaccel_context;

    VAPictureParameterBufferMPEG2 *pic_param;

    VAIQMatrixBufferMPEG2 *iq_matrix;

    int i;



    av_dlog(avctx, "vaapi_mpeg2_start_frame()\n");



    vactx->slice_param_size = sizeof(VASliceParameterBufferMPEG2);



    /* Fill in VAPictureParameterBufferMPEG2 */

    pic_param = ff_vaapi_alloc_pic_param(vactx, sizeof(VAPictureParameterBufferMPEG2));

    if (!pic_param)

        return -1;

    pic_param->horizontal_size                                  = s->width;

    pic_param->vertical_size                                    = s->height;

    pic_param->forward_reference_picture                        = VA_INVALID_ID;

    pic_param->backward_reference_picture                       = VA_INVALID_ID;

    pic_param->picture_coding_type                              = s->pict_type;

    pic_param->f_code                                           = mpeg2_get_f_code(s);

    pic_param->picture_coding_extension.value                   = 0; /* reset all bits */

    pic_param->picture_coding_extension.bits.intra_dc_precision = s->intra_dc_precision;

    pic_param->picture_coding_extension.bits.picture_structure  = s->picture_structure;

    pic_param->picture_coding_extension.bits.top_field_first    = s->top_field_first;

    pic_param->picture_coding_extension.bits.frame_pred_frame_dct = s->frame_pred_frame_dct;

    pic_param->picture_coding_extension.bits.concealment_motion_vectors = s->concealment_motion_vectors;

    pic_param->picture_coding_extension.bits.q_scale_type       = s->q_scale_type;

    pic_param->picture_coding_extension.bits.intra_vlc_format   = s->intra_vlc_format;

    pic_param->picture_coding_extension.bits.alternate_scan     = s->alternate_scan;

    pic_param->picture_coding_extension.bits.repeat_first_field = s->repeat_first_field;

    pic_param->picture_coding_extension.bits.progressive_frame  = s->progressive_frame;

    pic_param->picture_coding_extension.bits.is_first_field     = mpeg2_get_is_frame_start(s);



    switch (s->pict_type) {

    case AV_PICTURE_TYPE_B:

        pic_param->backward_reference_picture = ff_vaapi_get_surface_id(&s->next_picture.f);

        // fall-through

    case AV_PICTURE_TYPE_P:

        pic_param->forward_reference_picture = ff_vaapi_get_surface_id(&s->last_picture.f);

        break;

    }



    /* Fill in VAIQMatrixBufferMPEG2 */

    iq_matrix = ff_vaapi_alloc_iq_matrix(vactx, sizeof(VAIQMatrixBufferMPEG2));

    if (!iq_matrix)

        return -1;

    iq_matrix->load_intra_quantiser_matrix              = 1;

    iq_matrix->load_non_intra_quantiser_matrix          = 1;

    iq_matrix->load_chroma_intra_quantiser_matrix       = 1;

    iq_matrix->load_chroma_non_intra_quantiser_matrix   = 1;



    for (i = 0; i < 64; i++) {

        int n = s->dsp.idct_permutation[ff_zigzag_direct[i]];

        iq_matrix->intra_quantiser_matrix[i]            = s->intra_matrix[n];

        iq_matrix->non_intra_quantiser_matrix[i]        = s->inter_matrix[n];

        iq_matrix->chroma_intra_quantiser_matrix[i]     = s->chroma_intra_matrix[n];

        iq_matrix->chroma_non_intra_quantiser_matrix[i] = s->chroma_inter_matrix[n];

    }

    return 0;

}
