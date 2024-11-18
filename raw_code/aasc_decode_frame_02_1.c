static int aasc_decode_frame(AVCodecContext *avctx,
                              void *data, int *data_size,
                              AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    AascContext *s = avctx->priv_data;
    int compr, i, stride, psize;
    s->frame.reference = 3;
    s->frame.buffer_hints = FF_BUFFER_HINTS_VALID | FF_BUFFER_HINTS_PRESERVE | FF_BUFFER_HINTS_REUSABLE;
    if (avctx->reget_buffer(avctx, &s->frame)) {
        av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");
        return -1;
    compr = AV_RL32(buf);
    buf += 4;
    buf_size -= 4;
    psize = avctx->bits_per_coded_sample / 8;
    switch (avctx->codec_tag) {
    case MKTAG('A', 'A', 'S', '4'):
        bytestream2_init(&s->gb, buf - 4, buf_size + 4);
        ff_msrle_decode(avctx, (AVPicture*)&s->frame, 8, &s->gb);
        break;
    case MKTAG('A', 'A', 'S', 'C'):
    switch(compr){
    case 0:
        stride = (avctx->width * psize + psize) & ~psize;
        for(i = avctx->height - 1; i >= 0; i--){
            if(avctx->width * psize > buf_size){
                av_log(avctx, AV_LOG_ERROR, "Next line is beyond buffer bounds\n");
                break;
            memcpy(s->frame.data[0] + i*s->frame.linesize[0], buf, avctx->width * psize);
            buf += stride;
            buf_size -= stride;
        break;
    case 1:
        bytestream2_init(&s->gb, buf, buf_size);
        ff_msrle_decode(avctx, (AVPicture*)&s->frame, 8, &s->gb);
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Unknown compression type %d\n", compr);
        return -1;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Unknown FourCC: %X\n", avctx->codec_tag);
        return -1;
    if (avctx->pix_fmt == AV_PIX_FMT_PAL8)
        memcpy(s->frame.data[1], s->palette, s->palette_size);
    *data_size = sizeof(AVFrame);
    *(AVFrame*)data = s->frame;
    /* report that the buffer was completely consumed */
    return buf_size;