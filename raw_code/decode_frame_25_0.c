static int decode_frame(AVCodecContext *avctx,

                        void *data,

                        int *got_frame,

                        AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    DPXContext *const s = avctx->priv_data;

    AVFrame *picture  = data;

    AVFrame *const p = &s->picture;

    uint8_t *ptr[AV_NUM_DATA_POINTERS];



    unsigned int offset;

    int magic_num, endian;

    int x, y, i, ret;

    int w, h, bits_per_color, descriptor, elements, packing, total_size;



    unsigned int rgbBuffer = 0;

    int n_datum = 0;



    if (avpkt->size <= 1634) {

        av_log(avctx, AV_LOG_ERROR, "Packet too small for DPX header\n");

        return AVERROR_INVALIDDATA;

    }



    magic_num = AV_RB32(buf);

    buf += 4;



    /* Check if the files "magic number" is "SDPX" which means it uses

     * big-endian or XPDS which is for little-endian files */

    if (magic_num == AV_RL32("SDPX")) {

        endian = 0;

    } else if (magic_num == AV_RB32("SDPX")) {

        endian = 1;

    } else {

        av_log(avctx, AV_LOG_ERROR, "DPX marker not found\n");

        return AVERROR_INVALIDDATA;

    }



    offset = read32(&buf, endian);

    if (avpkt->size <= offset) {

        av_log(avctx, AV_LOG_ERROR, "Invalid data start offset\n");

        return AVERROR_INVALIDDATA;

    }

    // Need to end in 0x304 offset from start of file

    buf = avpkt->data + 0x304;

    w = read32(&buf, endian);

    h = read32(&buf, endian);

    if ((ret = av_image_check_size(w, h, 0, avctx)) < 0)

        return ret;



    if (w != avctx->width || h != avctx->height)

        avcodec_set_dimensions(avctx, w, h);



    // Need to end in 0x320 to read the descriptor

    buf += 20;

    descriptor = buf[0];



    // Need to end in 0x323 to read the bits per color

    buf += 3;

    avctx->bits_per_raw_sample =

    bits_per_color = buf[0];

    buf++;

    packing = *((uint16_t*)buf);



    buf += 824;

    avctx->sample_aspect_ratio.num = read32(&buf, endian);

    avctx->sample_aspect_ratio.den = read32(&buf, endian);

    if (avctx->sample_aspect_ratio.num > 0 && avctx->sample_aspect_ratio.den > 0)

        av_reduce(&avctx->sample_aspect_ratio.num, &avctx->sample_aspect_ratio.den,

                   avctx->sample_aspect_ratio.num,  avctx->sample_aspect_ratio.den,

                  0x10000);

    else

        avctx->sample_aspect_ratio = (AVRational){ 0, 1 };



    switch (descriptor) {

        case 51: // RGBA

            elements = 4;

            break;

        case 50: // RGB

            elements = 3;

            break;

        default:

            av_log(avctx, AV_LOG_ERROR, "Unsupported descriptor %d\n", descriptor);

            return AVERROR_INVALIDDATA;

    }



    switch (bits_per_color) {

        case 8:

            if (elements == 4) {

                avctx->pix_fmt = AV_PIX_FMT_RGBA;

            } else {

                avctx->pix_fmt = AV_PIX_FMT_RGB24;

            }

            total_size = avctx->width * avctx->height * elements;

            break;

        case 10:

            if (!packing) {

                av_log(avctx, AV_LOG_ERROR, "Packing to 32bit required\n");

                return -1;

            }

            avctx->pix_fmt = AV_PIX_FMT_GBRP10;

            total_size = (avctx->width * avctx->height * elements + 2) / 3 * 4;

            break;

        case 12:

            if (!packing) {

                av_log(avctx, AV_LOG_ERROR, "Packing to 16bit required\n");

                return -1;

            }

            if (endian) {

                avctx->pix_fmt = AV_PIX_FMT_GBRP12BE;

            } else {

                avctx->pix_fmt = AV_PIX_FMT_GBRP12LE;

            }

            total_size = 2 * avctx->width * avctx->height * elements;

            break;

        case 16:

            if (endian) {

                avctx->pix_fmt = elements == 4 ? AV_PIX_FMT_RGBA64BE : AV_PIX_FMT_RGB48BE;

            } else {

                avctx->pix_fmt = elements == 4 ? AV_PIX_FMT_RGBA64LE : AV_PIX_FMT_RGB48LE;

            }

            total_size = 2 * avctx->width * avctx->height * elements;

            break;

        default:

            av_log(avctx, AV_LOG_ERROR, "Unsupported color depth : %d\n", bits_per_color);

            return AVERROR_INVALIDDATA;

    }



    if (s->picture.data[0])

        avctx->release_buffer(avctx, &s->picture);

    if ((ret = ff_get_buffer(avctx, p)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    // Move pointer to offset from start of file

    buf =  avpkt->data + offset;



    for (i=0; i<AV_NUM_DATA_POINTERS; i++)

        ptr[i] = p->data[i];



    if (total_size > avpkt->size) {

        av_log(avctx, AV_LOG_ERROR, "Overread buffer. Invalid header?\n");

        return AVERROR_INVALIDDATA;

    }

    switch (bits_per_color) {

    case 10:

        for (x = 0; x < avctx->height; x++) {

            uint16_t *dst[3] = {(uint16_t*)ptr[0],

                                (uint16_t*)ptr[1],

                                (uint16_t*)ptr[2]};

            for (y = 0; y < avctx->width; y++) {

                *dst[2]++ = read10in32(&buf, &rgbBuffer,

                                       &n_datum, endian);

                *dst[0]++ = read10in32(&buf, &rgbBuffer,

                                       &n_datum, endian);

                *dst[1]++ = read10in32(&buf, &rgbBuffer,

                                       &n_datum, endian);

                // For 10 bit, ignore alpha

                if (elements == 4)

                    read10in32(&buf, &rgbBuffer,

                               &n_datum, endian);

            }

            for (i = 0; i < 3; i++)

                ptr[i] += p->linesize[i];

        }

        break;

    case 12:

        for (x = 0; x < avctx->height; x++) {

            uint16_t *dst[3] = {(uint16_t*)ptr[0],

                                (uint16_t*)ptr[1],

                                (uint16_t*)ptr[2]};

            for (y = 0; y < avctx->width; y++) {

                *dst[2] = *((uint16_t*)buf);

                *dst[2] = (*dst[2] >> 4) | (*dst[2] << 12);

                dst[2]++;

                buf += 2;

                *dst[0] = *((uint16_t*)buf);

                *dst[0] = (*dst[0] >> 4) | (*dst[0] << 12);

                dst[0]++;

                buf += 2;

                *dst[1] = *((uint16_t*)buf);

                *dst[1] = (*dst[1] >> 4) | (*dst[1] << 12);

                dst[1]++;

                buf += 2;

                // For 12 bit, ignore alpha

                if (elements == 4)

                    buf += 2;

            }

            for (i = 0; i < 3; i++)

                ptr[i] += p->linesize[i];

        }

        break;

    case 16:

        elements *= 2;

    case 8:

        for (x = 0; x < avctx->height; x++) {

            memcpy(ptr[0], buf, elements*avctx->width);

            ptr[0] += p->linesize[0];

            buf += elements*avctx->width;

        }

        break;

    }



    *picture   = s->picture;

    *got_frame = 1;



    return buf_size;

}
