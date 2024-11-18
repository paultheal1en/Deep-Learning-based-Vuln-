static int decode_frame(AVCodecContext *avctx,

                        void *data, int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    TiffContext *const s = avctx->priv_data;

    AVFrame *picture = data;

    AVFrame *const p = &s->picture;

    const uint8_t *orig_buf = buf, *end_buf = buf + buf_size;

    unsigned off;

    int id, le, ret;

    int i, j, entries;

    int stride;

    unsigned soff, ssize;

    uint8_t *dst;



    //parse image header

    if (end_buf - buf < 8)

        return AVERROR_INVALIDDATA;

    id = AV_RL16(buf);

    buf += 2;

    if (id == 0x4949)

        le = 1;

    else if (id == 0x4D4D)

        le = 0;

    else {

        av_log(avctx, AV_LOG_ERROR, "TIFF header not found\n");

        return -1;

    }

    s->le = le;

    s->invert = 0;

    s->compr = TIFF_RAW;

    s->fill_order = 0;

    free_geotags(s);

    /* free existing metadata */

    av_dict_free(&s->picture.metadata);



    // As TIFF 6.0 specification puts it "An arbitrary but carefully chosen number

    // that further identifies the file as a TIFF file"

    if (tget_short(&buf, le) != 42) {

        av_log(avctx, AV_LOG_ERROR,

               "The answer to life, universe and everything is not correct!\n");

        return -1;

    }

    // Reset these pointers so we can tell if they were set this frame

    s->stripsizes = s->stripdata = NULL;

    /* parse image file directory */

    off = tget_long(&buf, le);

    if (off >= UINT_MAX - 14 || end_buf - orig_buf < off + 14) {

        av_log(avctx, AV_LOG_ERROR, "IFD offset is greater than image size\n");

        return AVERROR_INVALIDDATA;

    }

    buf = orig_buf + off;

    entries = tget_short(&buf, le);

    for (i = 0; i < entries; i++) {

        if (tiff_decode_tag(s, orig_buf, buf, end_buf) < 0)

            return -1;

        buf += 12;

    }



    for (i = 0; i<s->geotag_count; i++) {

        const char *keyname = get_geokey_name(s->geotags[i].key);

        if (!keyname) {

            av_log(avctx, AV_LOG_WARNING, "Unknown or unsupported GeoTIFF key %d\n", s->geotags[i].key);

            continue;

        }

        if (get_geokey_type(s->geotags[i].key) != s->geotags[i].type) {

            av_log(avctx, AV_LOG_WARNING, "Type of GeoTIFF key %d is wrong\n", s->geotags[i].key);

            continue;

        }

        ret = av_dict_set(&s->picture.metadata, keyname, s->geotags[i].val, 0);

        if (ret<0) {

            av_log(avctx, AV_LOG_ERROR, "Writing metadata with key '%s' failed\n", keyname);

            return ret;

        }

    }



    if (!s->stripdata && !s->stripoff) {

        av_log(avctx, AV_LOG_ERROR, "Image data is missing\n");

        return -1;

    }

    /* now we have the data and may start decoding */

    if ((ret = init_image(s)) < 0)

        return ret;



    if (s->strips == 1 && !s->stripsize) {

        av_log(avctx, AV_LOG_WARNING, "Image data size missing\n");

        s->stripsize = buf_size - s->stripoff;

    }

    stride = p->linesize[0];

    dst = p->data[0];

    for (i = 0; i < s->height; i += s->rps) {

        if (s->stripsizes) {

            if (s->stripsizes >= end_buf)

                return AVERROR_INVALIDDATA;

            ssize = tget(&s->stripsizes, s->sstype, s->le);

        } else

            ssize = s->stripsize;



        if (s->stripdata) {

            if (s->stripdata >= end_buf)

                return AVERROR_INVALIDDATA;

            soff = tget(&s->stripdata, s->sot, s->le);

        } else

            soff = s->stripoff;



        if (soff > buf_size || ssize > buf_size - soff) {

            av_log(avctx, AV_LOG_ERROR, "Invalid strip size/offset\n");

            return -1;

        }

        if (tiff_unpack_strip(s, dst, stride, orig_buf + soff, ssize,

                              FFMIN(s->rps, s->height - i)) < 0)

            break;

        dst += s->rps * stride;

    }

    if (s->predictor == 2) {

        dst = p->data[0];

        soff = s->bpp >> 3;

        ssize = s->width * soff;

        if (s->avctx->pix_fmt == PIX_FMT_RGB48LE ||

            s->avctx->pix_fmt == PIX_FMT_RGBA64LE) {

            for (i = 0; i < s->height; i++) {

                for (j = soff; j < ssize; j += 2)

                    AV_WL16(dst + j, AV_RL16(dst + j) + AV_RL16(dst + j - soff));

                dst += stride;

            }

        } else if (s->avctx->pix_fmt == PIX_FMT_RGB48BE ||

                   s->avctx->pix_fmt == PIX_FMT_RGBA64BE) {

            for (i = 0; i < s->height; i++) {

                for (j = soff; j < ssize; j += 2)

                    AV_WB16(dst + j, AV_RB16(dst + j) + AV_RB16(dst + j - soff));

                dst += stride;

            }

        } else {

            for (i = 0; i < s->height; i++) {

                for (j = soff; j < ssize; j++)

                    dst[j] += dst[j - soff];

                dst += stride;

            }

        }

    }



    if (s->invert) {

        dst = s->picture.data[0];

        for (i = 0; i < s->height; i++) {

            for (j = 0; j < s->picture.linesize[0]; j++)

                dst[j] = (s->avctx->pix_fmt == PIX_FMT_PAL8 ? (1<<s->bpp) - 1 : 255) - dst[j];

            dst += s->picture.linesize[0];

        }

    }

    *picture   = s->picture;

    *data_size = sizeof(AVPicture);



    return buf_size;

}
