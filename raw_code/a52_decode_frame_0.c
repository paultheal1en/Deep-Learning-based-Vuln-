static int a52_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    AC3DecodeState *s = avctx->priv_data;

    uint8_t *buf_ptr;

    int flags, i, len;

    int sample_rate, bit_rate;

    short *out_samples = data;

    float level;

    static const int ac3_channels[8] = {

        2, 1, 2, 3, 3, 4, 4, 5

    };



    *data_size= 0;



    buf_ptr = buf;

    while (buf_size > 0) {

        len = s->inbuf_ptr - s->inbuf;

        if (s->frame_size == 0) {

            /* no header seen : find one. We need at least 7 bytes to parse it */

            len = HEADER_SIZE - len;

            if (len > buf_size)

                len = buf_size;

            memcpy(s->inbuf_ptr, buf_ptr, len);

            buf_ptr += len;

            s->inbuf_ptr += len;

            buf_size -= len;

            if ((s->inbuf_ptr - s->inbuf) == HEADER_SIZE) {

                len = s->a52_syncinfo(s->inbuf, &s->flags, &sample_rate, &bit_rate);

                if (len == 0) {

                    /* no sync found : move by one byte (inefficient, but simple!) */

                    memcpy(s->inbuf, s->inbuf + 1, HEADER_SIZE - 1);

                    s->inbuf_ptr--;

                } else {

                    s->frame_size = len;

                    /* update codec info */

                    avctx->sample_rate = sample_rate;

                    s->channels = ac3_channels[s->flags & 7];

                    if (s->flags & A52_LFE)

                        s->channels++;

                        avctx->channels = s->channels;

                    if (avctx->request_channels > 0 &&

                            avctx->request_channels <= 2 &&

                            avctx->request_channels < s->channels) {

                        avctx->channels = avctx->request_channels;

                    }

                    avctx->bit_rate = bit_rate;

                }

            }

        } else if (len < s->frame_size) {

            len = s->frame_size - len;

            if (len > buf_size)

                len = buf_size;



            memcpy(s->inbuf_ptr, buf_ptr, len);

            buf_ptr += len;

            s->inbuf_ptr += len;

            buf_size -= len;

        } else {

            flags = s->flags;

            if (avctx->channels == 1)

                flags = A52_MONO;

            else if (avctx->channels == 2)

                flags = A52_STEREO;

            else

                flags |= A52_ADJUST_LEVEL;

            level = 1;

            if (s->a52_frame(s->state, s->inbuf, &flags, &level, 384)) {

            fail:

                av_log(avctx, AV_LOG_ERROR, "Error decoding frame\n");

                s->inbuf_ptr = s->inbuf;

                s->frame_size = 0;

                continue;

            }

            for (i = 0; i < 6; i++) {

                if (s->a52_block(s->state))

                    goto fail;

                float_to_int(s->samples, out_samples + i * 256 * avctx->channels, avctx->channels);

            }

            s->inbuf_ptr = s->inbuf;

            s->frame_size = 0;

            *data_size = 6 * avctx->channels * 256 * sizeof(int16_t);

            break;

        }

    }

    return buf_ptr - buf;

}
