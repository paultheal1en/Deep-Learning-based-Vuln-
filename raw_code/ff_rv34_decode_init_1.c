av_cold int ff_rv34_decode_init(AVCodecContext *avctx)

{

    RV34DecContext *r = avctx->priv_data;

    MpegEncContext *s = &r->s;

    int ret;



    ff_MPV_decode_defaults(s);

    s->avctx      = avctx;

    s->out_format = FMT_H263;

    s->codec_id   = avctx->codec_id;



    s->width  = avctx->width;

    s->height = avctx->height;



    r->s.avctx = avctx;

    avctx->flags |= CODEC_FLAG_EMU_EDGE;

    r->s.flags |= CODEC_FLAG_EMU_EDGE;

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;

    avctx->has_b_frames = 1;

    s->low_delay = 0;



    if ((ret = ff_MPV_common_init(s)) < 0)

        return ret;



    ff_h264_pred_init(&r->h, AV_CODEC_ID_RV40, 8, 1);



#if CONFIG_RV30_DECODER

    if (avctx->codec_id == AV_CODEC_ID_RV30)

        ff_rv30dsp_init(&r->rdsp);

#endif

#if CONFIG_RV40_DECODER

    if (avctx->codec_id == AV_CODEC_ID_RV40)

        ff_rv40dsp_init(&r->rdsp);

#endif



    if ((ret = rv34_decoder_alloc(r)) < 0)

        return ret;



    if(!intra_vlcs[0].cbppattern[0].bits)

        rv34_init_tables();



    avctx->internal->allocate_progress = 1;



    return 0;

}
