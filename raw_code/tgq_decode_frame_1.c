static int tgq_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt){

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    const uint8_t *buf_start = buf;

    const uint8_t *buf_end = buf + buf_size;

    TgqContext *s = avctx->priv_data;

    int x,y;



    int big_endian = AV_RL32(&buf[4]) > 0x000FFFFF;

    buf += 8;



    if(8>buf_end-buf) {

        av_log(avctx, AV_LOG_WARNING, "truncated header\n");

        return -1;

    }

    s->width  = big_endian ? AV_RB16(&buf[0]) : AV_RL16(&buf[0]);

    s->height = big_endian ? AV_RB16(&buf[2]) : AV_RL16(&buf[2]);



    if (s->avctx->width!=s->width || s->avctx->height!=s->height) {

        avcodec_set_dimensions(s->avctx, s->width, s->height);

        if (s->frame.data[0])

            avctx->release_buffer(avctx, &s->frame);

    }

    tgq_calculate_qtable(s, buf[4]);

    buf += 8;



    if (!s->frame.data[0]) {

        s->frame.key_frame = 1;

        s->frame.pict_type = AV_PICTURE_TYPE_I;

        s->frame.buffer_hints = FF_BUFFER_HINTS_VALID;

        if (avctx->get_buffer(avctx, &s->frame)) {

            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            return -1;

        }

    }



    for (y=0; y<(avctx->height+15)/16; y++)

    for (x=0; x<(avctx->width+15)/16; x++)

        tgq_decode_mb(s, y, x, &buf, buf_end);



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->frame;



    return buf-buf_start;

}
