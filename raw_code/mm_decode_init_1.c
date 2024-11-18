static av_cold int mm_decode_init(AVCodecContext *avctx)
{
    MmContext *s = avctx->priv_data;
    s->avctx = avctx;
    avctx->pix_fmt = AV_PIX_FMT_PAL8;
    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);
    return 0;