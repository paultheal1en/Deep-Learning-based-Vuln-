static int g722_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                             const AVFrame *frame, int *got_packet_ptr)

{

    G722Context *c = avctx->priv_data;

    const int16_t *samples = (const int16_t *)frame->data[0];

    int nb_samples, out_size, ret;



    out_size = (frame->nb_samples + 1) / 2;

    if ((ret = ff_alloc_packet2(avctx, avpkt, out_size)))

        return ret;



    nb_samples = frame->nb_samples - (frame->nb_samples & 1);



    if (avctx->trellis)

        g722_encode_trellis(c, avctx->trellis, avpkt->data, nb_samples, samples);

    else

        g722_encode_no_trellis(c, avpkt->data, nb_samples, samples);



    /* handle last frame with odd frame_size */

    if (nb_samples < frame->nb_samples) {

        int16_t last_samples[2] = { samples[nb_samples], samples[nb_samples] };

        encode_byte(c, &avpkt->data[nb_samples >> 1], last_samples);

    }



    if (frame->pts != AV_NOPTS_VALUE)

        avpkt->pts = frame->pts - ff_samples_to_time_base(avctx, avctx->delay);

    *got_packet_ptr = 1;

    return 0;

}
