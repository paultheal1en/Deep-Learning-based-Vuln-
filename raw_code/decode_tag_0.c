static int decode_tag(AVCodecContext * avctx,

                      void *data, int *data_size,

                      AVPacket *avpkt) {

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    NellyMoserDecodeContext *s = avctx->priv_data;

    int blocks, i;

    int16_t* samples;

    *data_size = 0;

    samples = (int16_t*)data;



    if (buf_size < avctx->block_align)

        return buf_size;



    if (buf_size % 64) {

        av_log(avctx, AV_LOG_ERROR, "Tag size %d.\n", buf_size);

        return buf_size;

    }

    blocks = buf_size / 64;

    /* Normal numbers of blocks for sample rates:

     *  8000 Hz - 1

     * 11025 Hz - 2

     * 16000 Hz - 3

     * 22050 Hz - 4

     * 44100 Hz - 8

     */



    for (i=0 ; i<blocks ; i++) {

        nelly_decode_block(s, &buf[i*NELLY_BLOCK_LEN], s->float_buf);

        s->fmt_conv.float_to_int16(&samples[i*NELLY_SAMPLES], s->float_buf, NELLY_SAMPLES);

        *data_size += NELLY_SAMPLES*sizeof(int16_t);

    }



    return buf_size;

}
