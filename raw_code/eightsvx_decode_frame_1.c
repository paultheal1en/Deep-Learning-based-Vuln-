static int eightsvx_decode_frame(AVCodecContext *avctx, void *data,

                                 int *got_frame_ptr, AVPacket *avpkt)

{

    EightSvxContext *esc = avctx->priv_data;

    int n, out_data_size, ret;

    uint8_t *src, *dst;



    /* decode and interleave the first packet */

    if (!esc->samples && avpkt) {

        uint8_t *deinterleaved_samples, *p = NULL;



        esc->samples_size = !esc->table ?

            avpkt->size : avctx->channels + (avpkt->size-avctx->channels) * 2;

        if (!(esc->samples = av_malloc(esc->samples_size)))

            return AVERROR(ENOMEM);



        /* decompress */

        if (esc->table) {

            const uint8_t *buf = avpkt->data;

            uint8_t *dst;

            int buf_size = avpkt->size;

            int i, n = esc->samples_size;



            if (buf_size < 2) {

                av_log(avctx, AV_LOG_ERROR, "packet size is too small\n");

                return AVERROR(EINVAL);

            }

            if (!(deinterleaved_samples = av_mallocz(n)))

                return AVERROR(ENOMEM);

            dst = p = deinterleaved_samples;



            /* the uncompressed starting value is contained in the first byte */

            dst = deinterleaved_samples;

            for (i = 0; i < avctx->channels; i++) {

                delta_decode(dst, buf + 1, buf_size / avctx->channels - 1, buf[0], esc->table);

                buf += buf_size / avctx->channels;

                dst += n / avctx->channels - 1;

            }

        } else {

            deinterleaved_samples = avpkt->data;

        }



        if (avctx->channels == 2)

            interleave_stereo(esc->samples, deinterleaved_samples, esc->samples_size);

        else

            memcpy(esc->samples, deinterleaved_samples, esc->samples_size);

        av_freep(&p);

    }



    /* get output buffer */

    av_assert1(!(esc->samples_size % avctx->channels || esc->samples_idx % avctx->channels));

    esc->frame.nb_samples = FFMIN(MAX_FRAME_SIZE, esc->samples_size - esc->samples_idx)  / avctx->channels;

    if ((ret = avctx->get_buffer(avctx, &esc->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    *got_frame_ptr   = 1;

    *(AVFrame *)data = esc->frame;



    dst = esc->frame.data[0];

    src = esc->samples + esc->samples_idx;

    out_data_size = esc->frame.nb_samples * avctx->channels;

    for (n = out_data_size; n > 0; n--)

        *dst++ = *src++ + 128;

    esc->samples_idx += out_data_size;



    return esc->table ?

        (avctx->frame_number == 0)*2 + out_data_size / 2 :

        out_data_size;

}
