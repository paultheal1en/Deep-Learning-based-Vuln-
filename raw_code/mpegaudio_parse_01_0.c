static int mpegaudio_parse(AVCodecParserContext *s1,

                           AVCodecContext *avctx,

                           const uint8_t **poutbuf, int *poutbuf_size,

                           const uint8_t *buf, int buf_size)

{

    MpegAudioParseContext *s = s1->priv_data;

    int len, ret, sr;

    uint32_t header;

    const uint8_t *buf_ptr;



    *poutbuf = NULL;

    *poutbuf_size = 0;

    buf_ptr = buf;

    while (buf_size > 0) {

        len = s->inbuf_ptr - s->inbuf;

        if (s->frame_size == 0) {

            /* special case for next header for first frame in free

               format case (XXX: find a simpler method) */

            if (s->free_format_next_header != 0) {

                AV_WB32(s->inbuf, s->free_format_next_header);

                s->inbuf_ptr = s->inbuf + 4;

                s->free_format_next_header = 0;

                goto got_header;

            }

            /* no header seen : find one. We need at least MPA_HEADER_SIZE

               bytes to parse it */

            len = FFMIN(MPA_HEADER_SIZE - len, buf_size);

            if (len > 0) {

                memcpy(s->inbuf_ptr, buf_ptr, len);

                buf_ptr += len;

                buf_size -= len;

                s->inbuf_ptr += len;

            }

            if ((s->inbuf_ptr - s->inbuf) >= MPA_HEADER_SIZE) {

            got_header:

                header = AV_RB32(s->inbuf);



                ret = ff_mpa_decode_header(avctx, header, &sr);

                if (ret < 0) {

                    s->header_count= -2;

                    /* no sync found : move by one byte (inefficient, but simple!) */

                    memmove(s->inbuf, s->inbuf + 1, s->inbuf_ptr - s->inbuf - 1);

                    s->inbuf_ptr--;

                    dprintf(avctx, "skip %x\n", header);

                    /* reset free format frame size to give a chance

                       to get a new bitrate */

                    s->free_format_frame_size = 0;

                } else {

                    if((header&SAME_HEADER_MASK) != (s->header&SAME_HEADER_MASK) && s->header)

                        s->header_count= -3;

                    s->header= header;

                    s->header_count++;

                    s->frame_size = ret;



#if 0

                    /* free format: prepare to compute frame size */

                    if (ff_mpegaudio_decode_header(s, header) == 1) {

                        s->frame_size = -1;

                    }

#endif

                    if(s->header_count > 1)

                        avctx->sample_rate= sr;

                }

            }

        } else

#if 0

        if (s->frame_size == -1) {

            /* free format : find next sync to compute frame size */

            len = MPA_MAX_CODED_FRAME_SIZE - len;

            if (len > buf_size)

                len = buf_size;

            if (len == 0) {

                /* frame too long: resync */

                s->frame_size = 0;

                memmove(s->inbuf, s->inbuf + 1, s->inbuf_ptr - s->inbuf - 1);

                s->inbuf_ptr--;

            } else {

                uint8_t *p, *pend;

                uint32_t header1;

                int padding;



                memcpy(s->inbuf_ptr, buf_ptr, len);

                /* check for header */

                p = s->inbuf_ptr - 3;

                pend = s->inbuf_ptr + len - 4;

                while (p <= pend) {

                    header = AV_RB32(p);

                    header1 = AV_RB32(s->inbuf);

                    /* check with high probability that we have a

                       valid header */

                    if ((header & SAME_HEADER_MASK) ==

                        (header1 & SAME_HEADER_MASK)) {

                        /* header found: update pointers */

                        len = (p + 4) - s->inbuf_ptr;

                        buf_ptr += len;

                        buf_size -= len;

                        s->inbuf_ptr = p;

                        /* compute frame size */

                        s->free_format_next_header = header;

                        s->free_format_frame_size = s->inbuf_ptr - s->inbuf;

                        padding = (header1 >> 9) & 1;

                        if (s->layer == 1)

                            s->free_format_frame_size -= padding * 4;

                        else

                            s->free_format_frame_size -= padding;

                        dprintf(avctx, "free frame size=%d padding=%d\n",

                                s->free_format_frame_size, padding);

                        ff_mpegaudio_decode_header(s, header1);

                        goto next_data;

                    }

                    p++;

                }

                /* not found: simply increase pointers */

                buf_ptr += len;

                s->inbuf_ptr += len;

                buf_size -= len;

            }

        } else

#endif

        if (len < s->frame_size) {

            if (s->frame_size > MPA_MAX_CODED_FRAME_SIZE)

                s->frame_size = MPA_MAX_CODED_FRAME_SIZE;

            len = FFMIN(s->frame_size - len, buf_size);

            memcpy(s->inbuf_ptr, buf_ptr, len);

            buf_ptr += len;

            s->inbuf_ptr += len;

            buf_size -= len;

        }



        if(s->frame_size > 0 && buf_ptr - buf == s->inbuf_ptr - s->inbuf

           && buf_size + buf_ptr - buf >= s->frame_size){

            if(s->header_count > 0){

                *poutbuf = buf;

                *poutbuf_size = s->frame_size;

            }

            buf_ptr = buf + s->frame_size;

            s->inbuf_ptr = s->inbuf;

            s->frame_size = 0;

            break;

        }



        //    next_data:

        if (s->frame_size > 0 &&

            (s->inbuf_ptr - s->inbuf) >= s->frame_size) {

            if(s->header_count > 0){

                *poutbuf = s->inbuf;

                *poutbuf_size = s->inbuf_ptr - s->inbuf;

            }

            s->inbuf_ptr = s->inbuf;

            s->frame_size = 0;

            break;

        }

    }

    return buf_ptr - buf;

}
