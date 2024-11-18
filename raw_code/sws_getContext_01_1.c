struct SwsContext *sws_getContext(int srcW, int srcH, int srcFormat,

                                  int dstW, int dstH, int dstFormat,

                                  int flags, SwsFilter *srcFilter,

                                  SwsFilter *dstFilter, double *param)

{

    struct SwsContext *ctx;



    ctx = av_malloc(sizeof(struct SwsContext));

    if (ctx)

        ctx->av_class = av_mallocz(sizeof(AVClass));

    if (!ctx || !ctx->av_class) {

        av_log(NULL, AV_LOG_ERROR, "Cannot allocate a resampling context!\n");



        return NULL;

    }



    if ((srcH != dstH) || (srcW != dstW)) {

        if ((srcFormat != PIX_FMT_YUV420P) || (dstFormat != PIX_FMT_YUV420P)) {

            av_log(NULL, AV_LOG_INFO, "PIX_FMT_YUV420P will be used as an intermediate format for rescaling\n");

        }

        ctx->resampling_ctx = img_resample_init(dstW, dstH, srcW, srcH);

    } else {

        ctx->resampling_ctx = av_malloc(sizeof(ImgReSampleContext));

        ctx->resampling_ctx->iheight = srcH;

        ctx->resampling_ctx->iwidth = srcW;

        ctx->resampling_ctx->oheight = dstH;

        ctx->resampling_ctx->owidth = dstW;

    }

    ctx->src_pix_fmt = srcFormat;

    ctx->dst_pix_fmt = dstFormat;



    return ctx;

}
