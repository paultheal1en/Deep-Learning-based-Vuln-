static int smc_decode_frame(AVCodecContext *avctx,

                             void *data, int *got_frame,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    SmcContext *s = avctx->priv_data;

    const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, NULL);

    int ret;



    bytestream2_init(&s->gb, buf, buf_size);



    if ((ret = ff_reget_buffer(avctx, s->frame)) < 0)

        return ret;



    if (pal) {

        s->frame->palette_has_changed = 1;

        memcpy(s->pal, pal, AVPALETTE_SIZE);

    }



    smc_decode_stream(s);



    *got_frame      = 1;

    if ((ret = av_frame_ref(data, s->frame)) < 0)

        return ret;



    /* always report that the buffer was completely consumed */

    return buf_size;

}
