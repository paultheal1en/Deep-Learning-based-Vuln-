static av_cold int smc_decode_init(AVCodecContext *avctx)

{

    SmcContext *s = avctx->priv_data;



    s->avctx = avctx;

    avctx->pix_fmt = AV_PIX_FMT_PAL8;



    s->frame.data[0] = NULL;



    return 0;

}
