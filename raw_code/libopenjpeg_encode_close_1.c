static av_cold int libopenjpeg_encode_close(AVCodecContext *avctx)

{

    LibOpenJPEGContext *ctx = avctx->priv_data;



    opj_cio_close(ctx->stream);

    ctx->stream = NULL;

    opj_destroy_compress(ctx->compress);

    ctx->compress = NULL;

    opj_image_destroy(ctx->image);

    ctx->image = NULL;

    av_freep(&avctx->coded_frame);

    return 0;

}
