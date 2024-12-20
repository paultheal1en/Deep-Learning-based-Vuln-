static int decode_frame(AVCodecContext *avctx,

                        void *data, int *got_frame, AVPacket *avpkt)

{

    TiffContext *const s = avctx->priv_data;

    AVFrame *const p = data;

    ThreadFrame frame = { .f = data };

    unsigned off;

    int le, ret, plane, planes;

    int i, j, entries, stride;

    unsigned soff, ssize;

    uint8_t *dst;

    GetByteContext stripsizes;

    GetByteContext stripdata;



    bytestream2_init(&s->gb, avpkt->data, avpkt->size);



    // parse image header

    if ((ret = ff_tdecode_header(&s->gb, &le, &off))) {

        av_log(avctx, AV_LOG_ERROR, "Invalid TIFF header\n");

        return ret;

    } else if (off >= UINT_MAX - 14 || avpkt->size < off + 14) {

        av_log(avctx, AV_LOG_ERROR, "IFD offset is greater than image size\n");

        return AVERROR_INVALIDDATA;

    }

    s->le          = le;

    // TIFF_BPP is not a required tag and defaults to 1

    s->bppcount    = s->bpp = 1;

    s->photometric = TIFF_PHOTOMETRIC_NONE;

    s->compr       = TIFF_RAW;

    s->fill_order  = 0;

    free_geotags(s);



    // Reset these offsets so we can tell if they were set this frame

    s->stripsizesoff = s->strippos = 0;

    /* parse image file directory */

    bytestream2_seek(&s->gb, off, SEEK_SET);

    entries = ff_tget_short(&s->gb, le);

    if (bytestream2_get_bytes_left(&s->gb) < entries * 12)

        return AVERROR_INVALIDDATA;

    for (i = 0; i < entries; i++) {

        if ((ret = tiff_decode_tag(s, p)) < 0)

            return ret;

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

        ret = av_dict_set(avpriv_frame_get_metadatap(p), keyname, s->geotags[i].val, 0);

        if (ret<0) {

            av_log(avctx, AV_LOG_ERROR, "Writing metadata with key '%s' failed\n", keyname);

            return ret;

        }

    }



    if (!s->strippos && !s->stripoff) {

        av_log(avctx, AV_LOG_ERROR, "Image data is missing\n");

        return AVERROR_INVALIDDATA;

    }

    /* now we have the data and may start decoding */

    if ((ret = init_image(s, &frame)) < 0)

        return ret;



    if (s->strips == 1 && !s->stripsize) {

        av_log(avctx, AV_LOG_WARNING, "Image data size missing\n");

        s->stripsize = avpkt->size - s->stripoff;

    }



    if (s->stripsizesoff) {

        if (s->stripsizesoff >= (unsigned)avpkt->size)

            return AVERROR_INVALIDDATA;

        bytestream2_init(&stripsizes, avpkt->data + s->stripsizesoff,

                         avpkt->size - s->stripsizesoff);

    }

    if (s->strippos) {

        if (s->strippos >= (unsigned)avpkt->size)

            return AVERROR_INVALIDDATA;

        bytestream2_init(&stripdata, avpkt->data + s->strippos,

                         avpkt->size - s->strippos);

    }



    if (s->rps <= 0) {

        av_log(avctx, AV_LOG_ERROR, "rps %d invalid\n", s->rps);

        return AVERROR_INVALIDDATA;

    }



    planes = s->planar ? s->bppcount : 1;

    for (plane = 0; plane < planes; plane++) {

        stride = p->linesize[plane];

        dst = p->data[plane];

        for (i = 0; i < s->height; i += s->rps) {

            if (s->stripsizesoff)

                ssize = ff_tget(&stripsizes, s->sstype, le);

            else

                ssize = s->stripsize;



            if (s->strippos)

                soff = ff_tget(&stripdata, s->sot, le);

            else

                soff = s->stripoff;



            if (soff > avpkt->size || ssize > avpkt->size - soff) {

                av_log(avctx, AV_LOG_ERROR, "Invalid strip size/offset\n");

                return AVERROR_INVALIDDATA;

            }

            if ((ret = tiff_unpack_strip(s, p, dst, stride, avpkt->data + soff, ssize, i,

                                         FFMIN(s->rps, s->height - i))) < 0) {

                if (avctx->err_recognition & AV_EF_EXPLODE)

                    return ret;

                break;

            }

            dst += s->rps * stride;

        }

        if (s->predictor == 2) {

            if (s->photometric == TIFF_PHOTOMETRIC_YCBCR) {

                av_log(s->avctx, AV_LOG_ERROR, "predictor == 2 with YUV is unsupported");

                return AVERROR_PATCHWELCOME;

            }

            dst   = p->data[plane];

            soff  = s->bpp >> 3;

            if (s->planar)

                soff  = FFMAX(soff / s->bppcount, 1);

            ssize = s->width * soff;

            if (s->avctx->pix_fmt == AV_PIX_FMT_RGB48LE ||

                s->avctx->pix_fmt == AV_PIX_FMT_RGBA64LE ||

                s->avctx->pix_fmt == AV_PIX_FMT_GRAY16LE ||

                s->avctx->pix_fmt == AV_PIX_FMT_YA16LE ||

                s->avctx->pix_fmt == AV_PIX_FMT_GBRP16LE ||

                s->avctx->pix_fmt == AV_PIX_FMT_GBRAP16LE) {

                for (i = 0; i < s->height; i++) {

                    for (j = soff; j < ssize; j += 2)

                        AV_WL16(dst + j, AV_RL16(dst + j) + AV_RL16(dst + j - soff));

                    dst += stride;

                }

            } else if (s->avctx->pix_fmt == AV_PIX_FMT_RGB48BE ||

                       s->avctx->pix_fmt == AV_PIX_FMT_RGBA64BE ||

                       s->avctx->pix_fmt == AV_PIX_FMT_GRAY16BE ||

                       s->avctx->pix_fmt == AV_PIX_FMT_YA16BE ||

                       s->avctx->pix_fmt == AV_PIX_FMT_GBRP16BE ||

                       s->avctx->pix_fmt == AV_PIX_FMT_GBRAP16BE) {

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



        if (s->photometric == TIFF_PHOTOMETRIC_WHITE_IS_ZERO) {

            dst = p->data[plane];

            for (i = 0; i < s->height; i++) {

                for (j = 0; j < stride; j++)

                    dst[j] = (s->avctx->pix_fmt == AV_PIX_FMT_PAL8 ? (1<<s->bpp) - 1 : 255) - dst[j];

                dst += stride;

            }

        }

    }



    if (s->planar && s->bppcount > 2) {

        FFSWAP(uint8_t*, p->data[0],     p->data[2]);

        FFSWAP(int,      p->linesize[0], p->linesize[2]);

        FFSWAP(uint8_t*, p->data[0],     p->data[1]);

        FFSWAP(int,      p->linesize[0], p->linesize[1]);

    }



    *got_frame = 1;



    return avpkt->size;

}
