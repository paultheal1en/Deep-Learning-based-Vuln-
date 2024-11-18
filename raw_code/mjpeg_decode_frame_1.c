static int mjpeg_decode_frame(AVCodecContext *avctx, 

                              void *data, int *data_size,

                              UINT8 *buf, int buf_size)

{

    MJpegDecodeContext *s = avctx->priv_data;

    UINT8 *buf_end, *buf_ptr, *buf_start;

    int len, code, input_size, i;

    AVPicture *picture = data;

    unsigned int start_code;



    *data_size = 0;



    /* no supplementary picture */

    if (buf_size == 0)

        return 0;



    buf_ptr = buf;

    buf_end = buf + buf_size;

    while (buf_ptr < buf_end) {

        buf_start = buf_ptr;

        /* find start next marker */

        code = find_marker(&buf_ptr, buf_end, &s->header_state);

        /* copy to buffer */

        len = buf_ptr - buf_start;

        if (len + (s->buf_ptr - s->buffer) > s->buffer_size) {

            /* data too big : flush */

            s->buf_ptr = s->buffer;

            if (code > 0)

                s->start_code = code;

        } else {

            memcpy(s->buf_ptr, buf_start, len);

            s->buf_ptr += len;

            /* if we got FF 00, we copy FF to the stream to unescape FF 00 */

	    /* valid marker code is between 00 and ff - alex */

            if (code == 0 || code == 0xff) {

                s->buf_ptr--;

            } else if (code > 0) {

                /* prepare data for next start code */

                input_size = s->buf_ptr - s->buffer;

                start_code = s->start_code;

                s->buf_ptr = s->buffer;

                s->start_code = code;

                dprintf("marker=%x\n", start_code);

                switch(start_code) {

                case SOI:

                    /* nothing to do on SOI */

                    break;

                case DQT:

                    mjpeg_decode_dqt(s, s->buffer, input_size);

                    break;

                case DHT:

                    mjpeg_decode_dht(s, s->buffer, input_size);

                    break;

                case SOF0:

                    mjpeg_decode_sof0(s, s->buffer, input_size);

                    break;

                case SOS:

                    mjpeg_decode_sos(s, s->buffer, input_size);

                    if (s->start_code == EOI) {

                        int l;

                        if (s->interlaced) {

                            s->bottom_field ^= 1;

                            /* if not bottom field, do not output image yet */

                            if (s->bottom_field)

                                goto the_end;

                        }

                        for(i=0;i<3;i++) {

                            picture->data[i] = s->current_picture[i];

                            l = s->linesize[i];

                            if (s->interlaced)

                                l >>= 1;

                            picture->linesize[i] = l;

                        }

                        *data_size = sizeof(AVPicture);

                        avctx->height = s->height;

                        if (s->interlaced)

                            avctx->height *= 2;

                        avctx->width = s->width;

                        /* XXX: not complete test ! */

                        switch((s->h_count[0] << 4) | s->v_count[0]) {

                        case 0x11:

                            avctx->pix_fmt = PIX_FMT_YUV444P;

                            break;

                        case 0x21:

                            avctx->pix_fmt = PIX_FMT_YUV422P;

                            break;

                        default:

                        case 0x22:

                            avctx->pix_fmt = PIX_FMT_YUV420P;

                            break;

                        }

                        /* dummy quality */

                        /* XXX: infer it with matrix */

                        avctx->quality = 3; 

                        goto the_end;

                    }

                    break;

                }

            }

        }

    }

 the_end:

    return buf_ptr - buf;

}
