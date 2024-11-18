static int vdpau_mpeg4_start_frame(AVCodecContext *avctx,

                                   const uint8_t *buffer, uint32_t size)

{

    Mpeg4DecContext *ctx = avctx->priv_data;

    MpegEncContext * const s = &ctx->m;

    Picture *pic             = s->current_picture_ptr;

    struct vdpau_picture_context *pic_ctx = pic->hwaccel_picture_private;

    VdpPictureInfoMPEG4Part2 *info = &pic_ctx->info.mpeg4;

    VdpVideoSurface ref;

    int i;



    /* fill VdpPictureInfoMPEG4Part2 struct */

    info->forward_reference  = VDP_INVALID_HANDLE;

    info->backward_reference = VDP_INVALID_HANDLE;

    info->vop_coding_type    = 0;



    switch (s->pict_type) {

    case AV_PICTURE_TYPE_B:

        ref = ff_vdpau_get_surface_id(&s->next_picture.f);

        assert(ref != VDP_INVALID_HANDLE);

        info->backward_reference = ref;

        info->vop_coding_type    = 2;

        /* fall-through */

    case AV_PICTURE_TYPE_P:

        ref = ff_vdpau_get_surface_id(&s->last_picture.f);

        assert(ref != VDP_INVALID_HANDLE);

        info->forward_reference  = ref;

    }



    info->trd[0]                            = s->pp_time;

    info->trb[0]                            = s->pb_time;

    info->trd[1]                            = s->pp_field_time >> 1;

    info->trb[1]                            = s->pb_field_time >> 1;

    info->vop_time_increment_resolution     = s->avctx->time_base.den;

    info->vop_fcode_forward                 = s->f_code;

    info->vop_fcode_backward                = s->b_code;

    info->resync_marker_disable             = !ctx->resync_marker;

    info->interlaced                        = !s->progressive_sequence;

    info->quant_type                        = s->mpeg_quant;

    info->quarter_sample                    = s->quarter_sample;

    info->short_video_header                = avctx->codec->id == AV_CODEC_ID_H263;

    info->rounding_control                  = s->no_rounding;

    info->alternate_vertical_scan_flag      = s->alternate_scan;

    info->top_field_first                   = s->top_field_first;

    for (i = 0; i < 64; ++i) {

        info->intra_quantizer_matrix[i]     = s->intra_matrix[i];

        info->non_intra_quantizer_matrix[i] = s->inter_matrix[i];

    }



    ff_vdpau_common_start_frame(pic_ctx, buffer, size);

    return ff_vdpau_add_buffer(pic_ctx, buffer, size);

}
