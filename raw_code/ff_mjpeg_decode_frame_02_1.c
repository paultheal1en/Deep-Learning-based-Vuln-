int ff_mjpeg_decode_frame(AVCodecContext *avctx, void *data, int *got_frame,

                          AVPacket *avpkt)

{

    AVFrame     *frame = data;

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    MJpegDecodeContext *s = avctx->priv_data;

    const uint8_t *buf_end, *buf_ptr;

    const uint8_t *unescaped_buf_ptr;

    int hshift, vshift;

    int unescaped_buf_size;

    int start_code;

    int i, index;

    int ret = 0;

    int is16bit;



    av_dict_free(&s->exif_metadata);

    av_freep(&s->stereo3d);

    s->adobe_transform = -1;



    buf_ptr = buf;

    buf_end = buf + buf_size;

    while (buf_ptr < buf_end) {

        /* find start next marker */

        start_code = ff_mjpeg_find_marker(s, &buf_ptr, buf_end,

                                          &unescaped_buf_ptr,

                                          &unescaped_buf_size);

        /* EOF */

        if (start_code < 0) {

            break;

        } else if (unescaped_buf_size > INT_MAX / 8) {

            av_log(avctx, AV_LOG_ERROR,

                   "MJPEG packet 0x%x too big (%d/%d), corrupt data?\n",

                   start_code, unescaped_buf_size, buf_size);

            return AVERROR_INVALIDDATA;

        }

        av_log(avctx, AV_LOG_DEBUG, "marker=%x avail_size_in_buf=%"PTRDIFF_SPECIFIER"\n",

               start_code, buf_end - buf_ptr);



        ret = init_get_bits8(&s->gb, unescaped_buf_ptr, unescaped_buf_size);



        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "invalid buffer\n");

            goto fail;

        }



        s->start_code = start_code;

        if (s->avctx->debug & FF_DEBUG_STARTCODE)

            av_log(avctx, AV_LOG_DEBUG, "startcode: %X\n", start_code);



        /* process markers */

        if (start_code >= 0xd0 && start_code <= 0xd7)

            av_log(avctx, AV_LOG_DEBUG,

                   "restart marker: %d\n", start_code & 0x0f);

            /* APP fields */

        else if (start_code >= APP0 && start_code <= APP15)

            mjpeg_decode_app(s);

            /* Comment */

        else if (start_code == COM)

            mjpeg_decode_com(s);



        ret = -1;



        if (!CONFIG_JPEGLS_DECODER &&

            (start_code == SOF48 || start_code == LSE)) {

            av_log(avctx, AV_LOG_ERROR, "JPEG-LS support not enabled.\n");

            return AVERROR(ENOSYS);

        }



        if (avctx->skip_frame == AVDISCARD_ALL) {

            switch(start_code) {

            case SOF0:

            case SOF1:

            case SOF2:

            case SOF3:

            case SOF48:

            case SOI:

            case SOS:

            case EOI:

                break;

            default:

                goto skip;

            }

        }



        switch (start_code) {

        case SOI:

            s->restart_interval = 0;

            s->restart_count    = 0;

            /* nothing to do on SOI */

            break;

        case DQT:

            ff_mjpeg_decode_dqt(s);

            break;

        case DHT:

            if ((ret = ff_mjpeg_decode_dht(s)) < 0) {

                av_log(avctx, AV_LOG_ERROR, "huffman table decode error\n");

                goto fail;

            }

            break;

        case SOF0:

        case SOF1:

            s->lossless    = 0;

            s->ls          = 0;

            s->progressive = 0;

            if ((ret = ff_mjpeg_decode_sof(s)) < 0)

                goto fail;

            break;

        case SOF2:

            s->lossless    = 0;

            s->ls          = 0;

            s->progressive = 1;

            if ((ret = ff_mjpeg_decode_sof(s)) < 0)

                goto fail;

            break;

        case SOF3:

            s->avctx->properties |= FF_CODEC_PROPERTY_LOSSLESS;

            s->lossless    = 1;

            s->ls          = 0;

            s->progressive = 0;

            if ((ret = ff_mjpeg_decode_sof(s)) < 0)

                goto fail;

            break;

        case SOF48:

            s->avctx->properties |= FF_CODEC_PROPERTY_LOSSLESS;

            s->lossless    = 1;

            s->ls          = 1;

            s->progressive = 0;

            if ((ret = ff_mjpeg_decode_sof(s)) < 0)

                goto fail;

            break;

        case LSE:

            if (!CONFIG_JPEGLS_DECODER ||

                (ret = ff_jpegls_decode_lse(s)) < 0)

                goto fail;

            break;

        case EOI:

eoi_parser:

            s->cur_scan = 0;

            if (!s->got_picture) {

                av_log(avctx, AV_LOG_WARNING,

                       "Found EOI before any SOF, ignoring\n");

                break;

            }

            if (s->interlaced) {

                s->bottom_field ^= 1;

                /* if not bottom field, do not output image yet */

                if (s->bottom_field == !s->interlace_polarity)

                    break;

            }

            if (avctx->skip_frame == AVDISCARD_ALL) {

                s->got_picture = 0;

                goto the_end_no_picture;

            }

            if ((ret = av_frame_ref(frame, s->picture_ptr)) < 0)

                return ret;

            *got_frame = 1;

            s->got_picture = 0;



            if (!s->lossless) {

                int qp = FFMAX3(s->qscale[0],

                                s->qscale[1],

                                s->qscale[2]);

                int qpw = (s->width + 15) / 16;

                AVBufferRef *qp_table_buf = av_buffer_alloc(qpw);

                if (qp_table_buf) {

                    memset(qp_table_buf->data, qp, qpw);

                    av_frame_set_qp_table(data, qp_table_buf, 0, FF_QSCALE_TYPE_MPEG1);

                }



                if(avctx->debug & FF_DEBUG_QP)

                    av_log(avctx, AV_LOG_DEBUG, "QP: %d\n", qp);

            }



            goto the_end;

        case SOS:

            s->cur_scan++;

            if (avctx->skip_frame == AVDISCARD_ALL)

                break;



            if ((ret = ff_mjpeg_decode_sos(s, NULL, 0, NULL)) < 0 &&

                (avctx->err_recognition & AV_EF_EXPLODE))

                goto fail;

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

            av_log(avctx, AV_LOG_ERROR,

                   "mjpeg: unsupported coding type (%x)\n", start_code);

            break;

        }



skip:

        /* eof process start code */

        buf_ptr += (get_bits_count(&s->gb) + 7) / 8;

        av_log(avctx, AV_LOG_DEBUG,

               "marker parser used %d bytes (%d bits)\n",

               (get_bits_count(&s->gb) + 7) / 8, get_bits_count(&s->gb));

    }

    if (s->got_picture && s->cur_scan) {

        av_log(avctx, AV_LOG_WARNING, "EOI missing, emulating\n");

        goto eoi_parser;

    }

    av_log(avctx, AV_LOG_FATAL, "No JPEG data found in image\n");

    return AVERROR_INVALIDDATA;

fail:

    s->got_picture = 0;

    return ret;

the_end:



    is16bit = av_pix_fmt_desc_get(s->avctx->pix_fmt)->comp[0].step > 1;



    if (AV_RB32(s->upscale_h)) {

        int p;

        av_assert0(avctx->pix_fmt == AV_PIX_FMT_YUVJ444P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUV444P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVJ440P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUV440P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVA444P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVJ420P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUV420P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUV420P16||

                   avctx->pix_fmt == AV_PIX_FMT_YUVA420P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVA420P16||

                   avctx->pix_fmt == AV_PIX_FMT_GBRP     ||

                   avctx->pix_fmt == AV_PIX_FMT_GBRAP

                  );

        avcodec_get_chroma_sub_sample(s->avctx->pix_fmt, &hshift, &vshift);

        for (p = 0; p<4; p++) {

            uint8_t *line = s->picture_ptr->data[p];

            int w = s->width;

            int h = s->height;

            if (!s->upscale_h[p])

                continue;

            if (p==1 || p==2) {

                w = AV_CEIL_RSHIFT(w, hshift);

                h = AV_CEIL_RSHIFT(h, vshift);

            }

            if (s->upscale_v[p])

                h = (h+1)>>1;

            av_assert0(w > 0);

            for (i = 0; i < h; i++) {

                if (s->upscale_h[p] == 1) {

                    if (is16bit) ((uint16_t*)line)[w - 1] = ((uint16_t*)line)[(w - 1) / 2];

                    else                      line[w - 1] = line[(w - 1) / 2];

                    for (index = w - 2; index > 0; index--) {

                        if (is16bit)

                            ((uint16_t*)line)[index] = (((uint16_t*)line)[index / 2] + ((uint16_t*)line)[(index + 1) / 2]) >> 1;

                        else

                            line[index] = (line[index / 2] + line[(index + 1) / 2]) >> 1;

                    }

                } else if (s->upscale_h[p] == 2) {

                    if (is16bit) {

                        ((uint16_t*)line)[w - 1] = ((uint16_t*)line)[(w - 1) / 3];

                        if (w > 1)

                            ((uint16_t*)line)[w - 2] = ((uint16_t*)line)[w - 1];

                    } else {

                        line[w - 1] = line[(w - 1) / 3];

                        if (w > 1)

                            line[w - 2] = line[w - 1];

                    }

                    for (index = w - 3; index > 0; index--) {

                        line[index] = (line[index / 3] + line[(index + 1) / 3] + line[(index + 2) / 3] + 1) / 3;

                    }

                }

                line += s->linesize[p];

            }

        }

    }

    if (AV_RB32(s->upscale_v)) {

        int p;

        av_assert0(avctx->pix_fmt == AV_PIX_FMT_YUVJ444P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUV444P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVJ422P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUV422P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVJ420P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUV420P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUV440P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVJ440P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVA444P ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVA420P  ||

                   avctx->pix_fmt == AV_PIX_FMT_YUVA420P16||

                   avctx->pix_fmt == AV_PIX_FMT_GBRP     ||

                   avctx->pix_fmt == AV_PIX_FMT_GBRAP

                   );

        avcodec_get_chroma_sub_sample(s->avctx->pix_fmt, &hshift, &vshift);

        for (p = 0; p < 4; p++) {

            uint8_t *dst;

            int w = s->width;

            int h = s->height;

            if (!s->upscale_v[p])

                continue;

            if (p==1 || p==2) {

                w = AV_CEIL_RSHIFT(w, hshift);

                h = AV_CEIL_RSHIFT(h, vshift);

            }

            dst = &((uint8_t *)s->picture_ptr->data[p])[(h - 1) * s->linesize[p]];

            for (i = h - 1; i; i--) {

                uint8_t *src1 = &((uint8_t *)s->picture_ptr->data[p])[i / 2 * s->linesize[p]];

                uint8_t *src2 = &((uint8_t *)s->picture_ptr->data[p])[(i + 1) / 2 * s->linesize[p]];

                if (src1 == src2 || i == h - 1) {

                    memcpy(dst, src1, w);

                } else {

                    for (index = 0; index < w; index++)

                        dst[index] = (src1[index] + src2[index]) >> 1;

                }

                dst -= s->linesize[p];

            }

        }

    }

    if (s->flipped) {

        int j;

        avcodec_get_chroma_sub_sample(s->avctx->pix_fmt, &hshift, &vshift);

        for (index=0; index<4; index++) {

            uint8_t *dst = s->picture_ptr->data[index];

            int w = s->picture_ptr->width;

            int h = s->picture_ptr->height;

            if(index && index<3){

                w = AV_CEIL_RSHIFT(w, hshift);

                h = AV_CEIL_RSHIFT(h, vshift);

            }

            if(dst){

                uint8_t *dst2 = dst + s->picture_ptr->linesize[index]*(h-1);

                for (i=0; i<h/2; i++) {

                    for (j=0; j<w; j++)

                        FFSWAP(int, dst[j], dst2[j]);

                    dst  += s->picture_ptr->linesize[index];

                    dst2 -= s->picture_ptr->linesize[index];

                }

            }

        }

    }

    if (s->adobe_transform == 0 && s->avctx->pix_fmt == AV_PIX_FMT_GBRAP) {

        int w = s->picture_ptr->width;

        int h = s->picture_ptr->height;

        for (i=0; i<h; i++) {

            int j;

            uint8_t *dst[4];

            for (index=0; index<4; index++) {

                dst[index] =   s->picture_ptr->data[index]

                             + s->picture_ptr->linesize[index]*i;

            }

            for (j=0; j<w; j++) {

                int k = dst[3][j];

                int r = dst[0][j] * k;

                int g = dst[1][j] * k;

                int b = dst[2][j] * k;

                dst[0][j] = g*257 >> 16;

                dst[1][j] = b*257 >> 16;

                dst[2][j] = r*257 >> 16;

                dst[3][j] = 255;

            }

        }

    }

    if (s->adobe_transform == 2 && s->avctx->pix_fmt == AV_PIX_FMT_YUVA444P) {

        int w = s->picture_ptr->width;

        int h = s->picture_ptr->height;

        for (i=0; i<h; i++) {

            int j;

            uint8_t *dst[4];

            for (index=0; index<4; index++) {

                dst[index] =   s->picture_ptr->data[index]

                             + s->picture_ptr->linesize[index]*i;

            }

            for (j=0; j<w; j++) {

                int k = dst[3][j];

                int r = (255 - dst[0][j]) * k;

                int g = (128 - dst[1][j]) * k;

                int b = (128 - dst[2][j]) * k;

                dst[0][j] = r*257 >> 16;

                dst[1][j] = (g*257 >> 16) + 128;

                dst[2][j] = (b*257 >> 16) + 128;

                dst[3][j] = 255;

            }

        }

    }



    if (s->stereo3d) {

        AVStereo3D *stereo = av_stereo3d_create_side_data(data);

        if (stereo) {

            stereo->type  = s->stereo3d->type;

            stereo->flags = s->stereo3d->flags;

        }

        av_freep(&s->stereo3d);

    }



    av_dict_copy(avpriv_frame_get_metadatap(data), s->exif_metadata, 0);

    av_dict_free(&s->exif_metadata);



the_end_no_picture:

    av_log(avctx, AV_LOG_DEBUG, "decode frame unused %"PTRDIFF_SPECIFIER" bytes\n",

           buf_end - buf_ptr);

//  return buf_end - buf_ptr;

    return buf_ptr - buf;

}
