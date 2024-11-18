static int qdm2_decode_frame(AVCodecContext *avctx,

            void *data, int *data_size,

            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    QDM2Context *s = avctx->priv_data;

    int16_t *out = data;

    int i;



    if(!buf)

        return 0;

    if(buf_size < s->checksum_size)

        return -1;



    av_log(avctx, AV_LOG_DEBUG, "decode(%d): %p[%d] -> %p[%d]\n",

       buf_size, buf, s->checksum_size, data, *data_size);



    for (i = 0; i < 16; i++) {

        if (qdm2_decode(s, buf, out) < 0)

            return -1;

        out += s->channels * s->frame_size;

    }



    *data_size = (uint8_t*)out - (uint8_t*)data;



    return s->checksum_size;

}
