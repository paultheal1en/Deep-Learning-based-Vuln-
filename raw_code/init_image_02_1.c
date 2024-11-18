static int init_image(TiffContext *s, AVFrame *frame)
{
    int ret;
    switch (s->planar * 1000 + s->bpp * 10 + s->bppcount) {
    case 11:
        s->avctx->pix_fmt = AV_PIX_FMT_MONOBLACK;
        break;
    case 81:
        s->avctx->pix_fmt = s->palette_is_set ? AV_PIX_FMT_PAL8 : AV_PIX_FMT_GRAY8;
        break;
    case 243:
        s->avctx->pix_fmt = AV_PIX_FMT_RGB24;
        break;
    case 161:
        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_GRAY16LE : AV_PIX_FMT_GRAY16BE;
        break;
    case 162:
        s->avctx->pix_fmt = AV_PIX_FMT_YA8;
        break;
    case 322:
        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_YA16LE : AV_PIX_FMT_YA16BE;
        break;
    case 324:
        s->avctx->pix_fmt = AV_PIX_FMT_RGBA;
        break;
    case 483:
        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_RGB48LE : AV_PIX_FMT_RGB48BE;
        break;
    case 644:
        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_RGBA64LE : AV_PIX_FMT_RGBA64BE;
        break;
    case 1243:
        s->avctx->pix_fmt = AV_PIX_FMT_GBRP;
        break;
    case 1324:
        s->avctx->pix_fmt = AV_PIX_FMT_GBRAP;
        break;
    case 1483:
        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_GBRP16LE : AV_PIX_FMT_GBRP16BE;
        break;
    case 1644:
        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_GBRAP16LE : AV_PIX_FMT_GBRAP16BE;
        break;
    default:
               "This format is not supported (bpp=%d, bppcount=%d)\n",
    if (s->width != s->avctx->width || s->height != s->avctx->height) {
        ret = ff_set_dimensions(s->avctx, s->width, s->height);
        if (ret < 0)
            return ret;
    if ((ret = ff_get_buffer(s->avctx, frame, 0)) < 0) {
        av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed\n");
        return ret;
    if (s->avctx->pix_fmt == AV_PIX_FMT_PAL8) {
        memcpy(frame->data[1], s->palette, sizeof(s->palette));
    return 0;