static int tiff_unpack_strip(TiffContext *s, AVFrame *p, uint8_t *dst, int stride,

                             const uint8_t *src, int size, int strip_start, int lines)

{

    PutByteContext pb;

    int c, line, pixels, code, ret;

    const uint8_t *ssrc = src;

    int width = ((s->width * s->bpp) + 7) >> 3;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(p->format);

    int is_yuv = !(desc->flags & AV_PIX_FMT_FLAG_RGB) && desc->nb_components >= 2;



    if (s->planar)

        width /= s->bppcount;



    if (size <= 0)

        return AVERROR_INVALIDDATA;



    if (is_yuv) {

        int bytes_per_row = (((s->width - 1) / s->subsampling[0] + 1) * s->bpp *

                            s->subsampling[0] * s->subsampling[1] + 7) >> 3;

        av_fast_padded_malloc(&s->yuv_line, &s->yuv_line_size, bytes_per_row);

        if (s->yuv_line == NULL) {

            av_log(s->avctx, AV_LOG_ERROR, "Not enough memory\n");

            return AVERROR(ENOMEM);

        }

        dst = s->yuv_line;

        stride = 0;

        width = s->width * s->subsampling[1] + 2*(s->width / s->subsampling[0]);

        av_assert0(width <= bytes_per_row);

        av_assert0(s->bpp == 24);

    }



    if (s->compr == TIFF_DEFLATE || s->compr == TIFF_ADOBE_DEFLATE) {

#if CONFIG_ZLIB

        return tiff_unpack_zlib(s, p, dst, stride, src, size, width, lines,

                                strip_start, is_yuv);

#else

        av_log(s->avctx, AV_LOG_ERROR,

               "zlib support not enabled, "

               "deflate compression not supported\n");

        return AVERROR(ENOSYS);

#endif

    }

    if (s->compr == TIFF_LZMA) {

#if CONFIG_LZMA

        return tiff_unpack_lzma(s, p, dst, stride, src, size, width, lines,

                                strip_start, is_yuv);

#else

        av_log(s->avctx, AV_LOG_ERROR,

               "LZMA support not enabled\n");

        return AVERROR(ENOSYS);

#endif

    }

    if (s->compr == TIFF_LZW) {

        if (s->fill_order) {

            if ((ret = deinvert_buffer(s, src, size)) < 0)

                return ret;

            ssrc = src = s->deinvert_buf;

        }

        if (size > 1 && !src[0] && (src[1]&1)) {

            av_log(s->avctx, AV_LOG_ERROR, "Old style LZW is unsupported\n");

        }

        if ((ret = ff_lzw_decode_init(s->lzw, 8, src, size, FF_LZW_TIFF)) < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "Error initializing LZW decoder\n");

            return ret;

        }

        for (line = 0; line < lines; line++) {

            pixels = ff_lzw_decode(s->lzw, dst, width);

            if (pixels < width) {

                av_log(s->avctx, AV_LOG_ERROR, "Decoded only %i bytes of %i\n",

                       pixels, width);

                return AVERROR_INVALIDDATA;

            }

            if (s->bpp < 8 && s->avctx->pix_fmt == AV_PIX_FMT_PAL8)

                horizontal_fill(s->bpp, dst, 1, dst, 0, width, 0);

            if (is_yuv) {

                unpack_yuv(s, p, dst, strip_start + line);

                line += s->subsampling[1] - 1;

            }

            dst += stride;

        }

        return 0;

    }

    if (s->compr == TIFF_CCITT_RLE ||

        s->compr == TIFF_G3        ||

        s->compr == TIFF_G4) {

        if (is_yuv)

            return AVERROR_INVALIDDATA;



        return tiff_unpack_fax(s, dst, stride, src, size, width, lines);

    }



    bytestream2_init(&s->gb, src, size);

    bytestream2_init_writer(&pb, dst, is_yuv ? s->yuv_line_size : (stride * lines));



    for (line = 0; line < lines; line++) {

        if (src - ssrc > size) {

            av_log(s->avctx, AV_LOG_ERROR, "Source data overread\n");

            return AVERROR_INVALIDDATA;

        }



        if (bytestream2_get_bytes_left(&s->gb) == 0 || bytestream2_get_eof(&pb))

            break;

        bytestream2_seek_p(&pb, stride * line, SEEK_SET);

        switch (s->compr) {

        case TIFF_RAW:

            if (ssrc + size - src < width)

                return AVERROR_INVALIDDATA;



            if (!s->fill_order) {

                horizontal_fill(s->bpp * (s->avctx->pix_fmt == AV_PIX_FMT_PAL8),

                                dst, 1, src, 0, width, 0);

            } else {

                int i;

                for (i = 0; i < width; i++)

                    dst[i] = ff_reverse[src[i]];

            }

            src += width;

            break;

        case TIFF_PACKBITS:

            for (pixels = 0; pixels < width;) {

                if (ssrc + size - src < 2) {

                    av_log(s->avctx, AV_LOG_ERROR, "Read went out of bounds\n");

                    return AVERROR_INVALIDDATA;

                }

                code = s->fill_order ? (int8_t) ff_reverse[*src++]: (int8_t) *src++;

                if (code >= 0) {

                    code++;

                    if (pixels + code > width ||

                        ssrc + size - src < code) {

                        av_log(s->avctx, AV_LOG_ERROR,

                               "Copy went out of bounds\n");

                        return AVERROR_INVALIDDATA;

                    }

                    horizontal_fill(s->bpp * (s->avctx->pix_fmt == AV_PIX_FMT_PAL8),

                                    dst, 1, src, 0, code, pixels);

                    src    += code;

                    pixels += code;

                } else if (code != -128) { // -127..-1

                    code = (-code) + 1;

                    if (pixels + code > width) {

                        av_log(s->avctx, AV_LOG_ERROR,

                               "Run went out of bounds\n");

                        return AVERROR_INVALIDDATA;

                    }

                    c = *src++;

                    horizontal_fill(s->bpp * (s->avctx->pix_fmt == AV_PIX_FMT_PAL8),

                                    dst, 0, NULL, c, code, pixels);

                    pixels += code;

                }

            }

            if (s->fill_order) {

                int i;

                for (i = 0; i < width; i++)

                    dst[i] = ff_reverse[dst[i]];

            }

            break;

        }

        if (is_yuv) {

            unpack_yuv(s, p, dst, strip_start + line);

            line += s->subsampling[1] - 1;

        }

        dst += stride;

    }

    return 0;

}
