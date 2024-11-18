static int libgsm_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                               const AVFrame *frame, int *got_packet_ptr)

{

    int ret;

    gsm_signal *samples = (gsm_signal *)frame->data[0];

    struct gsm_state *state = avctx->priv_data;



    if ((ret = ff_alloc_packet2(avctx, avpkt, avctx->block_align)))

        return ret;



    switch(avctx->codec_id) {

    case AV_CODEC_ID_GSM:

        gsm_encode(state, samples, avpkt->data);

        break;

    case AV_CODEC_ID_GSM_MS:

        gsm_encode(state, samples,                  avpkt->data);

        gsm_encode(state, samples + GSM_FRAME_SIZE, avpkt->data + 32);

    }



    *got_packet_ptr = 1;

    return 0;

}
