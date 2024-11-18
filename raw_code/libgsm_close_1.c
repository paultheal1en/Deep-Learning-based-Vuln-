static av_cold int libgsm_close(AVCodecContext *avctx) {


    gsm_destroy(avctx->priv_data);

    avctx->priv_data = NULL;

    return 0;

}