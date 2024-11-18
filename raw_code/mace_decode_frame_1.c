static int mace_decode_frame(AVCodecContext *avctx, void *data,
                             int *got_frame_ptr, AVPacket *avpkt)
{
    AVFrame *frame     = data;
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    int16_t **samples;
    MACEContext *ctx = avctx->priv_data;
    int i, j, k, l, ret;
    int is_mace3 = (avctx->codec_id == AV_CODEC_ID_MACE3);
    /* get output buffer */
    frame->nb_samples = 3 * (buf_size << (1 - is_mace3)) / avctx->channels;
    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)
        return ret;
    samples = (int16_t **)frame->extended_data;
    for(i = 0; i < avctx->channels; i++) {
        int16_t *output = samples[i];
        for (j=0; j < buf_size / (avctx->channels << is_mace3); j++)
            for (k=0; k < (1 << is_mace3); k++) {
                uint8_t pkt = buf[(i << is_mace3) +
                                  (j*avctx->channels << is_mace3) + k];
                uint8_t val[2][3] = {{pkt >> 5, (pkt >> 3) & 3, pkt & 7 },
                                     {pkt & 7 , (pkt >> 3) & 3, pkt >> 5}};
                for (l=0; l < 3; l++) {
                    if (is_mace3)
                        chomp3(&ctx->chd[i], output, val[1][l], l);
                    else
                        chomp6(&ctx->chd[i], output, val[0][l], l);
                    output += 1 << (1-is_mace3);
    *got_frame_ptr = 1;
    return buf_size;