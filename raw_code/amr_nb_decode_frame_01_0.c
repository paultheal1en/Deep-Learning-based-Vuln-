static int amr_nb_decode_frame(AVCodecContext *avctx, void *data,

                               int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    AMRContext *s      = avctx->priv_data;

    static const uint8_t block_size[16] = { 12, 13, 15, 17, 19, 20, 26, 31, 5, 0, 0, 0, 0, 0, 0, 0 };

    enum Mode dec_mode;

    int packet_size;



    av_dlog(avctx, "amr_decode_frame buf=%p buf_size=%d frame_count=%d!!\n",

            buf, buf_size, avctx->frame_number);



    dec_mode    = (buf[0] >> 3) & 0x000F;

    packet_size = block_size[dec_mode] + 1;



    if (packet_size > buf_size) {

        av_log(avctx, AV_LOG_ERROR, "amr frame too short (%u, should be %u)\n",

               buf_size, packet_size);

        return AVERROR_INVALIDDATA;

    }



    av_dlog(avctx, "packet_size=%d buf= 0x%X %X %X %X\n",

              packet_size, buf[0], buf[1], buf[2], buf[3]);

    /* call decoder */

    Decoder_Interface_Decode(s->dec_state, buf, data, 0);

    *data_size = 160 * 2;



    return packet_size;

}
