static av_cold int hnm_decode_init(AVCodecContext *avctx)

{

    Hnm4VideoContext *hnm = avctx->priv_data;



    if (avctx->extradata_size < 1) {

        av_log(avctx, AV_LOG_ERROR,

               "Extradata missing, decoder requires version number\n");

        return AVERROR_INVALIDDATA;

    }



    hnm->version   = avctx->extradata[0];

    avctx->pix_fmt = AV_PIX_FMT_PAL8;

    hnm->width     = avctx->width;

    hnm->height    = avctx->height;

    hnm->buffer1   = av_mallocz(avctx->width * avctx->height);

    hnm->buffer2   = av_mallocz(avctx->width * avctx->height);

    hnm->processed = av_mallocz(avctx->width * avctx->height);



    if (!hnm->buffer1 || !hnm->buffer2 || !hnm->processed) {

        av_log(avctx, AV_LOG_ERROR, "av_mallocz() failed\n");

        av_freep(&hnm->buffer1);

        av_freep(&hnm->buffer2);

        av_freep(&hnm->processed);

        return AVERROR(ENOMEM);

    }



    hnm->current  = hnm->buffer1;

    hnm->previous = hnm->buffer2;



    return 0;

}
