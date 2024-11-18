static int encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                        const AVFrame *frame, int *got_packet_ptr)

{

    NellyMoserEncodeContext *s = avctx->priv_data;

    int ret;



    if (s->last_frame)

        return 0;



    memcpy(s->buf, s->buf + NELLY_SAMPLES, NELLY_BUF_LEN * sizeof(*s->buf));

    if (frame) {

        memcpy(s->buf + NELLY_BUF_LEN, frame->data[0],

               frame->nb_samples * sizeof(*s->buf));

        if (frame->nb_samples < NELLY_SAMPLES) {

            memset(s->buf + NELLY_BUF_LEN + avctx->frame_size, 0,

                   (NELLY_SAMPLES - frame->nb_samples) * sizeof(*s->buf));

            if (frame->nb_samples >= NELLY_BUF_LEN)

                s->last_frame = 1;

        }

        if ((ret = ff_af_queue_add(&s->afq, frame) < 0))

            return ret;

    } else {

        memset(s->buf + NELLY_BUF_LEN, 0, NELLY_SAMPLES * sizeof(*s->buf));

        s->last_frame = 1;

    }



    if ((ret = ff_alloc_packet(avpkt, NELLY_BLOCK_LEN))) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet\n");

        return ret;

    }

    encode_block(s, avpkt->data, avpkt->size);



    /* Get the next frame pts/duration */

    ff_af_queue_remove(&s->afq, avctx->frame_size, &avpkt->pts,

                       &avpkt->duration);



    *got_packet_ptr = 1;

    return 0;

}
