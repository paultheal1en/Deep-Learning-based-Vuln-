static int vmdaudio_decode_frame(AVCodecContext *avctx,

                                 void *data, int *data_size,

                                 AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    VmdAudioContext *s = avctx->priv_data;

    int block_type, silent_chunks;

    unsigned char *output_samples = (unsigned char *)data;



    if (buf_size < 16) {

        av_log(avctx, AV_LOG_WARNING, "skipping small junk packet\n");

        *data_size = 0;

        return buf_size;

    }



    block_type = buf[6];

    if (block_type < BLOCK_TYPE_AUDIO || block_type > BLOCK_TYPE_SILENCE) {

        av_log(avctx, AV_LOG_ERROR, "unknown block type: %d\n", block_type);

        return AVERROR(EINVAL);

    }

    buf      += 16;

    buf_size -= 16;



    silent_chunks = 0;

    if (block_type == BLOCK_TYPE_INITIAL) {

        uint32_t flags = AV_RB32(buf);

        silent_chunks  = av_popcount(flags);

        buf      += 4;

        buf_size -= 4;

    } else if (block_type == BLOCK_TYPE_SILENCE) {

        silent_chunks = 1;

        buf_size = 0; // should already be zero but set it just to be sure

    }



    /* ensure output buffer is large enough */

    if (*data_size < (avctx->block_align*silent_chunks + buf_size) * s->out_bps)

        return -1;



    *data_size = vmdaudio_loadsound(s, output_samples, buf, silent_chunks, buf_size);



    return avpkt->size;

}
