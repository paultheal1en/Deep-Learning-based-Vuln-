int ff_mjpeg_decode_frame(AVCodecContext *avctx,

                              void *data, int *data_size,

                              AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MJpegDecodeContext *s = avctx->priv_data;

    const uint8_t *buf_end, *buf_ptr;

    const uint8_t *unescaped_buf_ptr;

    int unescaped_buf_size;

    int start_code;

    AVFrame *picture = data;



    s->got_picture = 0; // picture from previous image can not be reused

    buf_ptr = buf;

    buf_end = buf + buf_size;

    while (buf_ptr < buf_end) {

        /* find start next marker */

        start_code = ff_mjpeg_find_marker(s, &buf_ptr, buf_end,

                                          &unescaped_buf_ptr, &unescaped_buf_size);

        {

            /* EOF */

            if (start_code < 0) {

                goto the_end;

            } else {

                av_log(avctx, AV_LOG_DEBUG, "marker=%x avail_size_in_buf=%td\n", start_code, buf_end - buf_ptr);



                init_get_bits(&s->gb, unescaped_buf_ptr, unescaped_buf_size*8);



                s->start_code = start_code;

                if(s->avctx->debug & FF_DEBUG_STARTCODE){

                    av_log(avctx, AV_LOG_DEBUG, "startcode: %X\n", start_code);

                }



                /* process markers */

                if (start_code >= 0xd0 && start_code <= 0xd7) {

                    av_log(avctx, AV_LOG_DEBUG, "restart marker: %d\n", start_code&0x0f);

                    /* APP fields */

                } else if (start_code >= APP0 && start_code <= APP15) {

                    mjpeg_decode_app(s);

                    /* Comment */

                } else if (start_code == COM){

                    mjpeg_decode_com(s);

                }



                switch(start_code) {

                case SOI:

                    s->restart_interval = 0;



                    s->restart_count = 0;

                    /* nothing to do on SOI */

                    break;

                case DQT:

                    ff_mjpeg_decode_dqt(s);

                    break;

                case DHT:

                    if(ff_mjpeg_decode_dht(s) < 0){

                        av_log(avctx, AV_LOG_ERROR, "huffman table decode error\n");

                        return -1;

                    }

                    break;

                case SOF0:

                case SOF1:

                    s->lossless=0;

                    s->ls=0;

                    s->progressive=0;

                    if (ff_mjpeg_decode_sof(s) < 0)

                        return -1;

                    break;

                case SOF2:

                    s->lossless=0;

                    s->ls=0;

                    s->progressive=1;

                    if (ff_mjpeg_decode_sof(s) < 0)

                        return -1;

                    break;

                case SOF3:

                    s->lossless=1;

                    s->ls=0;

                    s->progressive=0;

                    if (ff_mjpeg_decode_sof(s) < 0)

                        return -1;

                    break;

                case SOF48:

                    s->lossless=1;

                    s->ls=1;

                    s->progressive=0;

                    if (ff_mjpeg_decode_sof(s) < 0)

                        return -1;

                    break;

                case LSE:

                    if (!CONFIG_JPEGLS_DECODER || ff_jpegls_decode_lse(s) < 0)

                        return -1;

                    break;

                case EOI:

                    if ((s->buggy_avid && !s->interlaced) || s->restart_interval)

                        break;

eoi_parser:

                    s->cur_scan = 0;

                    if (!s->got_picture) {

                        av_log(avctx, AV_LOG_WARNING, "Found EOI before any SOF, ignoring\n");

                        break;

                    }

                    if (s->interlaced) {

                        s->bottom_field ^= 1;

                        /* if not bottom field, do not output image yet */

                        if (s->bottom_field == !s->interlace_polarity)

                            break;

                    }

                    *picture = *s->picture_ptr;

                    *data_size = sizeof(AVFrame);



                    if(!s->lossless){

                        picture->quality= FFMAX3(s->qscale[0], s->qscale[1], s->qscale[2]);

                        picture->qstride= 0;

                        picture->qscale_table= s->qscale_table;

                        memset(picture->qscale_table, picture->quality, (s->width+15)/16);

                        if(avctx->debug & FF_DEBUG_QP)

                            av_log(avctx, AV_LOG_DEBUG, "QP: %d\n", picture->quality);

                        picture->quality*= FF_QP2LAMBDA;

                    }



                    goto the_end;

                case SOS:

                    if (!s->got_picture) {

                        av_log(avctx, AV_LOG_WARNING, "Can not process SOS before SOF, skipping\n");

                        break;

                    }

                    if (ff_mjpeg_decode_sos(s, NULL, NULL) < 0 &&

                        avctx->error_recognition >= FF_ER_EXPLODE)

                      return AVERROR_INVALIDDATA;

                    /* buggy avid puts EOI every 10-20th frame */

                    /* if restart period is over process EOI */

                    if ((s->buggy_avid && !s->interlaced) || s->restart_interval)

                        goto eoi_parser;

                    break;

                case DRI:

                    mjpeg_decode_dri(s);

                    break;

                case SOF5:

                case SOF6:

                case SOF7:

                case SOF9:

                case SOF10:

                case SOF11:

                case SOF13:

                case SOF14:

                case SOF15:

                case JPG:

                    av_log(avctx, AV_LOG_ERROR, "mjpeg: unsupported coding type (%x)\n", start_code);

                    break;

//                default:

//                    printf("mjpeg: unsupported marker (%x)\n", start_code);

//                    break;

                }



                /* eof process start code */

                buf_ptr += (get_bits_count(&s->gb)+7)/8;

                av_log(avctx, AV_LOG_DEBUG, "marker parser used %d bytes (%d bits)\n",

                       (get_bits_count(&s->gb)+7)/8, get_bits_count(&s->gb));

            }

        }

    }

    if (s->got_picture) {

        av_log(avctx, AV_LOG_WARNING, "EOI missing, emulating\n");

        goto eoi_parser;

    }

    av_log(avctx, AV_LOG_FATAL, "No JPEG data found in image\n");

    return -1;

the_end:

    av_log(avctx, AV_LOG_DEBUG, "mjpeg decode frame unused %td bytes\n", buf_end - buf_ptr);

//    return buf_end - buf_ptr;

    return buf_ptr - buf;

}
