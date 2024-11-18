static int pixlet_decode_frame(AVCodecContext *avctx, void *data,
                               int *got_frame, AVPacket *avpkt)
{
    PixletContext *ctx = avctx->priv_data;
    int i, w, h, width, height, ret, version;
    AVFrame *p = data;
    ThreadFrame frame = { .f = data };
    uint32_t pktsize;
    bytestream2_init(&ctx->gb, avpkt->data, avpkt->size);
    pktsize = bytestream2_get_be32(&ctx->gb);
    if (pktsize <= 44 || pktsize - 4 > bytestream2_get_bytes_left(&ctx->gb)) {
        av_log(avctx, AV_LOG_ERROR, "Invalid packet size %"PRIu32"\n", pktsize);
    }
    version = bytestream2_get_le32(&ctx->gb);
    if (version != 1)
        avpriv_request_sample(avctx, "Version %d", version);
    bytestream2_skip(&ctx->gb, 4);
    if (bytestream2_get_be32(&ctx->gb) != 1)
    bytestream2_skip(&ctx->gb, 4);
    width  = bytestream2_get_be32(&ctx->gb);
    height = bytestream2_get_be32(&ctx->gb);
    w = FFALIGN(width,  1 << (NB_LEVELS + 1));
    h = FFALIGN(height, 1 << (NB_LEVELS + 1));
    ctx->levels = bytestream2_get_be32(&ctx->gb);
    if (ctx->levels != NB_LEVELS)
    ctx->depth = bytestream2_get_be32(&ctx->gb);
    if (ctx->depth < 8 || ctx->depth > 15) {
        avpriv_request_sample(avctx, "Depth %d", ctx->depth);
    }
    ret = ff_set_dimensions(avctx, w, h);
    if (ret < 0)
        return ret;
    avctx->width  = width;
    avctx->height = height;
    if (ctx->w != w || ctx->h != h) {
        free_buffers(avctx);
        ctx->w = w;
        ctx->h = h;
        ret = init_decoder(avctx);
        if (ret < 0) {
            free_buffers(avctx);
            ctx->w = 0;
            ctx->h = 0;
            return ret;
        }
    }
    bytestream2_skip(&ctx->gb, 8);
    p->pict_type = AV_PICTURE_TYPE_I;
    p->key_frame = 1;
    p->color_range = AVCOL_RANGE_JPEG;
    ret = ff_thread_get_buffer(avctx, &frame, 0);
    if (ret < 0)
        return ret;
    for (i = 0; i < 3; i++) {
        ret = decode_plane(avctx, i, avpkt, frame.f);
        if (ret < 0)
            return ret;
        if (avctx->flags & AV_CODEC_FLAG_GRAY)
            break;
    }
    postprocess_luma(frame.f, ctx->w, ctx->h, ctx->depth);
    postprocess_chroma(frame.f, ctx->w >> 1, ctx->h >> 1, ctx->depth);
    *got_frame = 1;
    return pktsize;
}