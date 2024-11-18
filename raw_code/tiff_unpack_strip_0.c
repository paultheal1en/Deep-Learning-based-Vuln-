static int tiff_unpack_strip(TiffContext *s, uint8_t* dst, int stride, const uint8_t *src, int size, int lines){

    int c, line, pixels, code;

    const uint8_t *ssrc = src;

    int width = ((s->width * s->bpp) + 7) >> 3;

#if CONFIG_ZLIB

    uint8_t *zbuf; unsigned long outlen;



    if(s->compr == TIFF_DEFLATE || s->compr == TIFF_ADOBE_DEFLATE){

        int ret;

        outlen = width * lines;

        zbuf = av_malloc(outlen);

        ret = tiff_uncompress(zbuf, &outlen, src, size);

        if(ret != Z_OK){

            av_log(s->avctx, AV_LOG_ERROR, "Uncompressing failed (%lu of %lu) with error %d\n", outlen, (unsigned long)width * lines, ret);

            av_free(zbuf);

            return -1;

        }

        src = zbuf;

        for(line = 0; line < lines; line++){

            memcpy(dst, src, width);

            dst += stride;

            src += width;

        }

        av_free(zbuf);

        return 0;

    }

#endif

    if(s->compr == TIFF_LZW){

        if(ff_lzw_decode_init(s->lzw, 8, src, size, FF_LZW_TIFF) < 0){

            av_log(s->avctx, AV_LOG_ERROR, "Error initializing LZW decoder\n");

            return -1;

        }

    }

    if(s->compr == TIFF_CCITT_RLE || s->compr == TIFF_G3 || s->compr == TIFF_G4){

        int i, ret = 0;

        uint8_t *src2 = av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE);



        if(!src2 || (unsigned)size + FF_INPUT_BUFFER_PADDING_SIZE < (unsigned)size){

            av_log(s->avctx, AV_LOG_ERROR, "Error allocating temporary buffer\n");

            return -1;

        }

        if(s->fax_opts & 2){

            av_log(s->avctx, AV_LOG_ERROR, "Uncompressed fax mode is not supported (yet)\n");

            av_free(src2);

            return -1;

        }

        if(!s->fill_order){

            memcpy(src2, src, size);

        }else{

            for(i = 0; i < size; i++)

                src2[i] = av_reverse[src[i]];

        }

        memset(src2+size, 0, FF_INPUT_BUFFER_PADDING_SIZE);

        switch(s->compr){

        case TIFF_CCITT_RLE:

        case TIFF_G3:

        case TIFF_G4:

            ret = ff_ccitt_unpack(s->avctx, src2, size, dst, lines, stride, s->compr, s->fax_opts);

            break;

        }

        av_free(src2);

        return ret;

    }

    for(line = 0; line < lines; line++){

        if(src - ssrc > size){

            av_log(s->avctx, AV_LOG_ERROR, "Source data overread\n");

            return -1;

        }

        switch(s->compr){

        case TIFF_RAW:

            if (ssrc + size - src < width)

                return AVERROR_INVALIDDATA;

            if (!s->fill_order) {

                memcpy(dst, src, width);

            } else {

                int i;

                for (i = 0; i < width; i++)

                    dst[i] = av_reverse[src[i]];

            }

            src += width;

            break;

        case TIFF_PACKBITS:

            for(pixels = 0; pixels < width;){

                code = (int8_t)*src++;

                if(code >= 0){

                    code++;

                    if(pixels + code > width){

                        av_log(s->avctx, AV_LOG_ERROR, "Copy went out of bounds\n");

                        return -1;

                    }

                    memcpy(dst + pixels, src, code);

                    src += code;

                    pixels += code;

                }else if(code != -128){ // -127..-1

                    code = (-code) + 1;

                    if(pixels + code > width){

                        av_log(s->avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                        return -1;

                    }

                    c = *src++;

                    memset(dst + pixels, c, code);

                    pixels += code;

                }

            }

            break;

        case TIFF_LZW:

            pixels = ff_lzw_decode(s->lzw, dst, width);

            if(pixels < width){

                av_log(s->avctx, AV_LOG_ERROR, "Decoded only %i bytes of %i\n", pixels, width);

                return -1;

            }

            break;

        }

        dst += stride;

    }

    return 0;

}
