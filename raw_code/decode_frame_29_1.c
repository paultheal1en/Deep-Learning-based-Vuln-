static int decode_frame(AVCodecContext *avctx, void *data,

                        int *got_frame_ptr, AVPacket *avpkt)

{

    BinkAudioContext *s = avctx->priv_data;

    AVFrame *frame      = data;

    GetBitContext *gb = &s->gb;

    int ret, consumed = 0;



    if (!get_bits_left(gb)) {

        uint8_t *buf;

        /* handle end-of-stream */

        if (!avpkt->size) {

            *got_frame_ptr = 0;

            return 0;

        }

        if (avpkt->size < 4) {

            av_log(avctx, AV_LOG_ERROR, "Packet is too small\n");

            return AVERROR_INVALIDDATA;

        }

        buf = av_realloc(s->packet_buffer, avpkt->size + FF_INPUT_BUFFER_PADDING_SIZE);

        if (!buf)

            return AVERROR(ENOMEM);


        s->packet_buffer = buf;

        memcpy(s->packet_buffer, avpkt->data, avpkt->size);

        if ((ret = init_get_bits8(gb, s->packet_buffer, avpkt->size)) < 0)

            return ret;

        consumed = avpkt->size;



        /* skip reported size */

        skip_bits_long(gb, 32);

    }



    /* get output buffer */

    frame->nb_samples = s->frame_len;

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)

        return ret;



    if (decode_block(s, (float **)frame->extended_data,

                     avctx->codec->id == AV_CODEC_ID_BINKAUDIO_DCT)) {

        av_log(avctx, AV_LOG_ERROR, "Incomplete packet\n");

        return AVERROR_INVALIDDATA;

    }

    get_bits_align32(gb);



    frame->nb_samples = s->block_size / avctx->channels;

    *got_frame_ptr    = 1;



    return consumed;

}