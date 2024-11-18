static int aacPlus_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                                const AVFrame *frame, int *got_packet)

{

    aacPlusAudioContext *s = avctx->priv_data;

    int32_t *input_buffer = (int32_t *)frame->data[0];

    int ret;



    if ((ret = ff_alloc_packet2(avctx, pkt, s->max_output_bytes)))

        return ret;



    pkt->size = aacplusEncEncode(s->aacplus_handle, input_buffer,

                                 s->samples_input, pkt->data, pkt->size);

    *got_packet   = 1;

    pkt->pts      = frame->pts;

    return 0;

}
