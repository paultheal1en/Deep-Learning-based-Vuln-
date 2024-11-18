static int msvideo1_decode_frame(AVCodecContext *avctx,
                                void *data, int *got_frame,
                                AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    Msvideo1Context *s = avctx->priv_data;
    int ret;
    s->buf = buf;
    s->size = buf_size;
    if ((ret = ff_reget_buffer(avctx, s->frame)) < 0)
        return ret;
    if (s->mode_8bit) {
        int size;
        const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, &size);
        if (pal && size == AVPALETTE_SIZE) {
            memcpy(s->pal, pal, AVPALETTE_SIZE);
            s->frame->palette_has_changed = 1;
        } else if (pal) {
            av_log(avctx, AV_LOG_ERROR, "Palette size %d is wrong\n", size);
    if (s->mode_8bit)
        msvideo1_decode_8bit(s);
    else
        msvideo1_decode_16bit(s);
    if ((ret = av_frame_ref(data, s->frame)) < 0)
        return ret;
    *got_frame      = 1;
    /* report that the buffer was completely consumed */
    return buf_size;