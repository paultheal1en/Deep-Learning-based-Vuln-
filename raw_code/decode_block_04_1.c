static int decode_block(AVCodecContext *avctx, void *tdata,

                        int jobnr, int threadnr)

{

    EXRContext *s = avctx->priv_data;

    AVFrame *const p = s->picture;

    EXRThreadData *td = &s->thread_data[threadnr];

    const uint8_t *channel_buffer[4] = { 0 };

    const uint8_t *buf = s->buf;

    uint64_t line_offset, uncompressed_size;

    uint16_t *ptr_x;

    uint8_t *ptr;

    uint32_t data_size;

    uint64_t line, col = 0;

    uint64_t tile_x, tile_y, tile_level_x, tile_level_y;

    const uint8_t *src;

    int axmax = (avctx->width - (s->xmax + 1)) * 2 * s->desc->nb_components; /* nb pixel to add at the right of the datawindow */

    int bxmin = s->xmin * 2 * s->desc->nb_components; /* nb pixel to add at the left of the datawindow */

    int i, x, buf_size = s->buf_size;

    int c, rgb_channel_count;

    float one_gamma = 1.0f / s->gamma;

    avpriv_trc_function trc_func = avpriv_get_trc_function_from_trc(s->apply_trc_type);

    int ret;



    line_offset = AV_RL64(s->gb.buffer + jobnr * 8);



    if (s->is_tile) {

        if (line_offset > buf_size - 20)

            return AVERROR_INVALIDDATA;



        src  = buf + line_offset + 20;



        tile_x = AV_RL32(src - 20);

        tile_y = AV_RL32(src - 16);

        tile_level_x = AV_RL32(src - 12);

        tile_level_y = AV_RL32(src - 8);



        data_size = AV_RL32(src - 4);

        if (data_size <= 0 || data_size > buf_size)

            return AVERROR_INVALIDDATA;



        if (tile_level_x || tile_level_y) { /* tile level, is not the full res level */

            avpriv_report_missing_feature(s->avctx, "Subres tile before full res tile");

            return AVERROR_PATCHWELCOME;

        }



        if (s->xmin || s->ymin) {

            avpriv_report_missing_feature(s->avctx, "Tiles with xmin/ymin");

            return AVERROR_PATCHWELCOME;

        }



        line = s->tile_attr.ySize * tile_y;

        col = s->tile_attr.xSize * tile_x;



        if (line < s->ymin || line > s->ymax ||

            col  < s->xmin || col  > s->xmax)

            return AVERROR_INVALIDDATA;



        td->ysize = FFMIN(s->tile_attr.ySize, s->ydelta - tile_y * s->tile_attr.ySize);

        td->xsize = FFMIN(s->tile_attr.xSize, s->xdelta - tile_x * s->tile_attr.xSize);



        if (col) { /* not the first tile of the line */

            bxmin = 0; /* doesn't add pixel at the left of the datawindow */

        }



        if ((col + td->xsize) != s->xdelta)/* not the last tile of the line */

            axmax = 0; /* doesn't add pixel at the right of the datawindow */



        td->channel_line_size = td->xsize * s->current_channel_offset;/* uncompress size of one line */

        uncompressed_size = td->channel_line_size * (uint64_t)td->ysize;/* uncompress size of the block */

    } else {

        if (line_offset > buf_size - 8)

            return AVERROR_INVALIDDATA;



        src  = buf + line_offset + 8;

        line = AV_RL32(src - 8);



        if (line < s->ymin || line > s->ymax)

            return AVERROR_INVALIDDATA;



        data_size = AV_RL32(src - 4);

        if (data_size <= 0 || data_size > buf_size)

            return AVERROR_INVALIDDATA;



        td->ysize          = FFMIN(s->scan_lines_per_block, s->ymax - line + 1); /* s->ydelta - line ?? */

        td->xsize          = s->xdelta;



        td->channel_line_size = td->xsize * s->current_channel_offset;/* uncompress size of one line */

        uncompressed_size = td->channel_line_size * (uint64_t)td->ysize;/* uncompress size of the block */



        if ((s->compression == EXR_RAW && (data_size != uncompressed_size ||

                                           line_offset > buf_size - uncompressed_size)) ||

            (s->compression != EXR_RAW && (data_size > uncompressed_size ||

                                           line_offset > buf_size - data_size))) {

            return AVERROR_INVALIDDATA;

        }

    }



    if (data_size < uncompressed_size || s->is_tile) { /* td->tmp is use for tile reorganization */

        av_fast_padded_malloc(&td->tmp, &td->tmp_size, uncompressed_size);

        if (!td->tmp)

            return AVERROR(ENOMEM);

    }



    if (data_size < uncompressed_size) {

        av_fast_padded_malloc(&td->uncompressed_data,

                              &td->uncompressed_size, uncompressed_size + 64);/* Force 64 padding for AVX2 reorder_pixels dst */



        if (!td->uncompressed_data)

            return AVERROR(ENOMEM);



        ret = AVERROR_INVALIDDATA;

        switch (s->compression) {

        case EXR_ZIP1:

        case EXR_ZIP16:

            ret = zip_uncompress(s, src, data_size, uncompressed_size, td);

            break;

        case EXR_PIZ:

            ret = piz_uncompress(s, src, data_size, uncompressed_size, td);

            break;

        case EXR_PXR24:

            ret = pxr24_uncompress(s, src, data_size, uncompressed_size, td);

            break;

        case EXR_RLE:

            ret = rle_uncompress(s, src, data_size, uncompressed_size, td);

            break;

        case EXR_B44:

        case EXR_B44A:

            ret = b44_uncompress(s, src, data_size, uncompressed_size, td);

            break;

        }

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "decode_block() failed.\n");

            return ret;

        }

        src = td->uncompressed_data;

    }



    if (!s->is_luma) {

        channel_buffer[0] = src + td->xsize * s->channel_offsets[0];

        channel_buffer[1] = src + td->xsize * s->channel_offsets[1];

        channel_buffer[2] = src + td->xsize * s->channel_offsets[2];

        rgb_channel_count = 3;

    } else { /* put y data in the first channel_buffer */

        channel_buffer[0] = src + td->xsize * s->channel_offsets[1];

        rgb_channel_count = 1;

    }

    if (s->channel_offsets[3] >= 0)

        channel_buffer[3] = src + td->xsize * s->channel_offsets[3];



    ptr = p->data[0] + line * p->linesize[0] + (col * s->desc->nb_components * 2);



    for (i = 0;

         i < td->ysize; i++, ptr += p->linesize[0]) {



        const uint8_t * a;

        const uint8_t *rgb[3];



        for (c = 0; c < rgb_channel_count; c++){

            rgb[c] = channel_buffer[c];

        }



        if (channel_buffer[3])

            a = channel_buffer[3];



        ptr_x = (uint16_t *) ptr;



        // Zero out the start if xmin is not 0

        memset(ptr_x, 0, bxmin);

        ptr_x += s->xmin * s->desc->nb_components;



        if (s->pixel_type == EXR_FLOAT) {

            // 32-bit

            if (trc_func) {

                for (x = 0; x < td->xsize; x++) {

                    union av_intfloat32 t;



                    for (c = 0; c < rgb_channel_count; c++) {

                        t.i = bytestream_get_le32(&rgb[c]);

                        t.f = trc_func(t.f);

                        *ptr_x++ = exr_flt2uint(t.i);

                    }

                    if (channel_buffer[3])

                        *ptr_x++ = exr_flt2uint(bytestream_get_le32(&a));

                }

            } else {

                for (x = 0; x < td->xsize; x++) {

                    union av_intfloat32 t;

                    int c;



                    for (c = 0; c < rgb_channel_count; c++) {

                        t.i = bytestream_get_le32(&rgb[c]);

                        if (t.f > 0.0f)  /* avoid negative values */

                            t.f = powf(t.f, one_gamma);

                        *ptr_x++ = exr_flt2uint(t.i);

                    }



                    if (channel_buffer[3])

                        *ptr_x++ = exr_flt2uint(bytestream_get_le32(&a));

                }

            }

        } else if (s->pixel_type == EXR_HALF) {

            // 16-bit

            for (x = 0; x < td->xsize; x++) {

                int c;

                for (c = 0; c < rgb_channel_count; c++) {

                    *ptr_x++ = s->gamma_table[bytestream_get_le16(&rgb[c])];

                }



                if (channel_buffer[3])

                    *ptr_x++ = exr_halflt2uint(bytestream_get_le16(&a));

            }

        } else if (s->pixel_type == EXR_UINT) {

            for (x = 0; x < td->xsize; x++) {

                for (c = 0; c < rgb_channel_count; c++) {

                    *ptr_x++ = bytestream_get_le32(&rgb[c]) >> 16;

                }



                if (channel_buffer[3])

                    *ptr_x++ = bytestream_get_le32(&a) >> 16;

            }

        }



        // Zero out the end if xmax+1 is not w

        memset(ptr_x, 0, axmax);



        channel_buffer[0] += td->channel_line_size;

        channel_buffer[1] += td->channel_line_size;

        channel_buffer[2] += td->channel_line_size;

        if (channel_buffer[3])

            channel_buffer[3] += td->channel_line_size;

    }



    return 0;

}
