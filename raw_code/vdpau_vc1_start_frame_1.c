static int vdpau_vc1_start_frame(AVCodecContext *avctx,

                                 const uint8_t *buffer, uint32_t size)

{

    VC1Context * const v  = avctx->priv_data;

    MpegEncContext * const s = &v->s;

    Picture *pic          = s->current_picture_ptr;

    struct vdpau_picture_context *pic_ctx = pic->hwaccel_picture_private;

    VdpPictureInfoVC1 *info = &pic_ctx->info.vc1;

    VdpVideoSurface ref;



    /*  fill LvPictureInfoVC1 struct */

    info->forward_reference  = VDP_INVALID_HANDLE;

    info->backward_reference = VDP_INVALID_HANDLE;



    switch (s->pict_type) {

    case AV_PICTURE_TYPE_B:

        ref = ff_vdpau_get_surface_id(&s->next_picture.f);

        assert(ref != VDP_INVALID_HANDLE);

        info->backward_reference = ref;

        /* fall-through */

    case AV_PICTURE_TYPE_P:

        ref = ff_vdpau_get_surface_id(&s->last_picture.f);

        assert(ref != VDP_INVALID_HANDLE);

        info->forward_reference  = ref;

    }



    info->slice_count       = 0;

    if (v->bi_type)

        info->picture_type  = 4;

    else

        info->picture_type  = s->pict_type - 1 + s->pict_type / 3;



    info->frame_coding_mode = v->fcm ? (v->fcm + 1) : 0;

    info->postprocflag      = v->postprocflag;

    info->pulldown          = v->broadcast;

    info->interlace         = v->interlace;

    info->tfcntrflag        = v->tfcntrflag;

    info->finterpflag       = v->finterpflag;

    info->psf               = v->psf;

    info->dquant            = v->dquant;

    info->panscan_flag      = v->panscanflag;

    info->refdist_flag      = v->refdist_flag;

    info->quantizer         = v->quantizer_mode;

    info->extended_mv       = v->extended_mv;

    info->extended_dmv      = v->extended_dmv;

    info->overlap           = v->overlap;

    info->vstransform       = v->vstransform;

    info->loopfilter        = v->s.loop_filter;

    info->fastuvmc          = v->fastuvmc;

    info->range_mapy_flag   = v->range_mapy_flag;

    info->range_mapy        = v->range_mapy;

    info->range_mapuv_flag  = v->range_mapuv_flag;

    info->range_mapuv       = v->range_mapuv;

    /* Specific to simple/main profile only */

    info->multires          = v->multires;

    info->syncmarker        = v->resync_marker;

    info->rangered          = v->rangered | (v->rangeredfrm << 1);

    info->maxbframes        = v->s.max_b_frames;

    info->deblockEnable     = v->postprocflag & 1;

    info->pquant            = v->pq;



    return ff_vdpau_common_start_frame(pic_ctx, buffer, size);

}
