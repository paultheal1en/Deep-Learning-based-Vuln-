static av_cold int tdsc_init(AVCodecContext *avctx)

{

    TDSCContext *ctx = avctx->priv_data;

    const AVCodec *codec;

    int ret;



    avctx->pix_fmt = AV_PIX_FMT_BGR24;



    /* These needs to be set to estimate buffer and frame size */

    if (!(avctx->width && avctx->height)) {

        av_log(avctx, AV_LOG_ERROR, "Video size not set.\n");

        return AVERROR_INVALIDDATA;

    }



    /* This value should be large enough for a RAW-only frame plus headers */

    ctx->deflatelen = avctx->width * avctx->height * (3 + 1);

    ret = av_reallocp(&ctx->deflatebuffer, ctx->deflatelen);

    if (ret < 0)

        return ret;



    /* Allocate reference and JPEG frame */

    ctx->refframe = av_frame_alloc();

    ctx->jpgframe = av_frame_alloc();

    if (!ctx->refframe || !ctx->jpgframe)

        return AVERROR(ENOMEM);



    /* Prepare everything needed for JPEG decoding */

    codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);

    if (!codec)

        return AVERROR_BUG;

    ctx->jpeg_avctx = avcodec_alloc_context3(codec);

    if (!ctx->jpeg_avctx)

        return AVERROR(ENOMEM);

    ctx->jpeg_avctx->flags = avctx->flags;

    ctx->jpeg_avctx->flags2 = avctx->flags2;

    ctx->jpeg_avctx->dct_algo = avctx->dct_algo;

    ctx->jpeg_avctx->idct_algo = avctx->idct_algo;;

    ret = avcodec_open2(ctx->jpeg_avctx, codec, NULL);

    if (ret < 0)

        return ret;



    /* Set the output pixel format on the reference frame */

    ctx->refframe->format = avctx->pix_fmt;



    return 0;

}
