static int atrac3_decode_frame(AVCodecContext *avctx,

            void *data, int *data_size,

            AVPacket *avpkt) {

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    ATRAC3Context *q = avctx->priv_data;

    int result = 0;

    const uint8_t* databuf;

    float *samples = data;



    if (buf_size < avctx->block_align) {

        av_log(avctx, AV_LOG_ERROR,

               "Frame too small (%d bytes). Truncated file?\n", buf_size);

        *data_size = 0;

        return buf_size;

    }



    /* Check if we need to descramble and what buffer to pass on. */

    if (q->scrambled_stream) {

        decode_bytes(buf, q->decoded_bytes_buffer, avctx->block_align);

        databuf = q->decoded_bytes_buffer;

    } else {

        databuf = buf;

    }



    result = decodeFrame(q, databuf, q->channels == 2 ? q->outSamples : &samples);



    if (result != 0) {

        av_log(NULL,AV_LOG_ERROR,"Frame decoding error!\n");

        return -1;

    }



    /* interleave */

    if (q->channels == 2) {

        q->fmt_conv.float_interleave(samples, (const float **)q->outSamples,

                                     1024, 2);

    }

    *data_size = 1024 * q->channels * av_get_bytes_per_sample(avctx->sample_fmt);



    return avctx->block_align;

}
