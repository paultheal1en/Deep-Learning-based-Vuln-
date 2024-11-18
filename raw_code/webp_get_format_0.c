static enum AVPixelFormat webp_get_format(AVCodecContext *avctx,

                                          const enum AVPixelFormat *formats)

{

    WebPContext *s = avctx->priv_data;

    if (s->has_alpha)

        return AV_PIX_FMT_YUVA420P;

    else

        return AV_PIX_FMT_YUV420P;

}
