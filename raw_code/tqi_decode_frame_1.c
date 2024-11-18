static int tqi_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    const uint8_t *buf_end = buf+buf_size;

    TqiContext *t = avctx->priv_data;

    MpegEncContext *s = &t->s;



    s->width  = AV_RL16(&buf[0]);

    s->height = AV_RL16(&buf[2]);

    tqi_calculate_qtable(s, buf[4]);

    buf += 8;



    if (t->frame.data[0])

        avctx->release_buffer(avctx, &t->frame);



    if (s->avctx->width!=s->width || s->avctx->height!=s->height)

        avcodec_set_dimensions(s->avctx, s->width, s->height);



    if(avctx->get_buffer(avctx, &t->frame) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    av_fast_malloc(&t->bitstream_buf, &t->bitstream_buf_size, (buf_end-buf) + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!t->bitstream_buf)

        return AVERROR(ENOMEM);

    s->dsp.bswap_buf(t->bitstream_buf, (const uint32_t*)buf, (buf_end-buf)/4);

    init_get_bits(&s->gb, t->bitstream_buf, 8*(buf_end-buf));



    s->last_dc[0] = s->last_dc[1] = s->last_dc[2] = 0;

    for (s->mb_y=0; s->mb_y<(avctx->height+15)/16; s->mb_y++)

    for (s->mb_x=0; s->mb_x<(avctx->width+15)/16; s->mb_x++)

    {

        if(tqi_decode_mb(s, t->block) < 0)

            break;

        tqi_idct_put(t, t->block);

    }



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = t->frame;

    return buf_size;

}
