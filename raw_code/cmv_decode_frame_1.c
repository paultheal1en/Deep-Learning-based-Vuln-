static int cmv_decode_frame(AVCodecContext *avctx,
                            void *data, int *data_size,
                            AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    CmvContext *s = avctx->priv_data;
    const uint8_t *buf_end = buf + buf_size;
    if (AV_RL32(buf)==MVIh_TAG||AV_RB32(buf)==MVIh_TAG) {
        cmv_process_header(s, buf+EA_PREAMBLE_SIZE, buf_end);
        return buf_size;
    }
    if (av_image_check_size(s->width, s->height, 0, s->avctx))
        return -1;
    /* shuffle */
    if (s->last2_frame.data[0])
        avctx->release_buffer(avctx, &s->last2_frame);
    FFSWAP(AVFrame, s->last_frame, s->last2_frame);
    FFSWAP(AVFrame, s->frame, s->last_frame);
    s->frame.reference = 1;
    s->frame.buffer_hints = FF_BUFFER_HINTS_VALID;
    if (avctx->get_buffer(avctx, &s->frame)<0) {
        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");
        return -1;
    }
    memcpy(s->frame.data[1], s->palette, AVPALETTE_SIZE);
    buf += EA_PREAMBLE_SIZE;
    if ((buf[0]&1)) {  // subtype
        cmv_decode_inter(s, buf+2, buf_end);
        s->frame.key_frame = 0;
        s->frame.pict_type = AV_PICTURE_TYPE_P;
    }else{
        s->frame.key_frame = 1;
        s->frame.pict_type = AV_PICTURE_TYPE_I;
        cmv_decode_intra(s, buf+2, buf_end);
    }
    *data_size = sizeof(AVFrame);
    *(AVFrame*)data = s->frame;
    return buf_size;
}