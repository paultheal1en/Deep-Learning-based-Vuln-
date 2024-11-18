static av_cold int mss2_decode_init(AVCodecContext *avctx)

{

    MSS2Context * const ctx = avctx->priv_data;

    MSS12Context *c = &ctx->c;

    int ret;

    c->avctx = avctx;

    avctx->coded_frame = &ctx->pic;

    if (ret = ff_mss12_decode_init(c, 1, &ctx->sc[0], &ctx->sc[1]))

        return ret;

    c->pal_stride   = c->mask_stride;

    c->pal_pic      = av_malloc(c->pal_stride * avctx->height);

    c->last_pal_pic = av_malloc(c->pal_stride * avctx->height);

    if (!c->pal_pic || !c->last_pal_pic) {

        mss2_decode_end(avctx);

        return AVERROR(ENOMEM);

    }

    if (ret = wmv9_init(avctx)) {

        mss2_decode_end(avctx);

        return ret;

    }

    ff_mss2dsp_init(&ctx->dsp);



    avctx->pix_fmt = c->free_colours == 127 ? AV_PIX_FMT_RGB555

                                            : AV_PIX_FMT_RGB24;



    return 0;

}
