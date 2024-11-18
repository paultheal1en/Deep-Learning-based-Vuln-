static int smc_decode_init(AVCodecContext *avctx)

{

    SmcContext *s = avctx->priv_data;



    s->avctx = avctx;

    avctx->pix_fmt = PIX_FMT_PAL8;

    dsputil_init(&s->dsp, avctx);



    s->frame.data[0] = NULL;



    return 0;

}
