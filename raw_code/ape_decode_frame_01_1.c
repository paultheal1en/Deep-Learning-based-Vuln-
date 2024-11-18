static int ape_decode_frame(AVCodecContext * avctx,
                            void *data, int *data_size,
                            const uint8_t * buf, int buf_size)
{
    APEContext *s = avctx->priv_data;
    int16_t *samples = data;
    int nblocks;
    int i, n;
    int blockstodecode;
    int bytes_used;
    if (buf_size == 0 && !s->samples) {
        *data_size = 0;
        return 0;
    /* should not happen but who knows */
    if (BLOCKS_PER_LOOP * 2 * avctx->channels > *data_size) {
        av_log (avctx, AV_LOG_ERROR, "Packet size is too big to be handled in lavc! (max is %d where you have %d)\n", *data_size, s->samples * 2 * avctx->channels);
    if(!s->samples){
        s->data = av_realloc(s->data, (buf_size + 3) & ~3);
        s->dsp.bswap_buf((uint32_t*)s->data, (const uint32_t*)buf, buf_size >> 2);
        s->ptr = s->last_ptr = s->data;
        s->data_end = s->data + buf_size;
        nblocks = s->samples = bytestream_get_be32(&s->ptr);
        n =  bytestream_get_be32(&s->ptr);
        if(n < 0 || n > 3){
            av_log(avctx, AV_LOG_ERROR, "Incorrect offset passed\n");
            s->data = NULL;
        s->ptr += n;
        s->currentframeblocks = nblocks;
        buf += 4;
        if (s->samples <= 0) {
            *data_size = 0;
            return buf_size;
        memset(s->decoded0,  0, sizeof(s->decoded0));
        memset(s->decoded1,  0, sizeof(s->decoded1));
        /* Initialize the frame decoder */
        init_frame_decoder(s);
    if (!s->data) {
        *data_size = 0;
        return buf_size;
    nblocks = s->samples;
    blockstodecode = FFMIN(BLOCKS_PER_LOOP, nblocks);
    if ((s->channels == 1) || (s->frameflags & APE_FRAMECODE_PSEUDO_STEREO))
        ape_unpack_mono(s, blockstodecode);
    else
        ape_unpack_stereo(s, blockstodecode);
    for (i = 0; i < blockstodecode; i++) {
        *samples++ = s->decoded0[i];
        if(s->channels == 2)
            *samples++ = s->decoded1[i];
    s->samples -= blockstodecode;
    *data_size = blockstodecode * 2 * s->channels;
    bytes_used = s->samples ? s->ptr - s->last_ptr : buf_size;
    s->last_ptr = s->ptr;
    return bytes_used;