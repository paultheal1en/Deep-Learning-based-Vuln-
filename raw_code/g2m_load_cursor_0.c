static int g2m_load_cursor(AVCodecContext *avctx, G2MContext *c,

                           GetByteContext *gb)

{

    int i, j, k;

    uint8_t *dst;

    uint32_t bits;

    uint32_t cur_size, cursor_w, cursor_h, cursor_stride;

    uint32_t cursor_hot_x, cursor_hot_y;

    int cursor_fmt;

    uint8_t *tmp;



    cur_size      = bytestream2_get_be32(gb);

    cursor_w      = bytestream2_get_byte(gb);

    cursor_h      = bytestream2_get_byte(gb);

    cursor_hot_x  = bytestream2_get_byte(gb);

    cursor_hot_y  = bytestream2_get_byte(gb);

    cursor_fmt    = bytestream2_get_byte(gb);



    cursor_stride = FFALIGN(cursor_w, c->cursor_fmt==1 ? 32 : 1) * 4;



    if (cursor_w < 1 || cursor_w > 256 ||

        cursor_h < 1 || cursor_h > 256) {

        av_log(avctx, AV_LOG_ERROR, "Invalid cursor dimensions %dx%d\n",

               cursor_w, cursor_h);

        return AVERROR_INVALIDDATA;

    }

    if (cursor_hot_x > cursor_w || cursor_hot_y > cursor_h) {

        av_log(avctx, AV_LOG_WARNING, "Invalid hotspot position %d,%d\n",

               cursor_hot_x, cursor_hot_y);

        cursor_hot_x = FFMIN(cursor_hot_x, cursor_w - 1);

        cursor_hot_y = FFMIN(cursor_hot_y, cursor_h - 1);

    }

    if (cur_size - 9 > bytestream2_get_bytes_left(gb) ||

        c->cursor_w * c->cursor_h / 4 > cur_size) {

        av_log(avctx, AV_LOG_ERROR, "Invalid cursor data size %d/%d\n",

               cur_size, bytestream2_get_bytes_left(gb));

        return AVERROR_INVALIDDATA;

    }

    if (cursor_fmt != 1 && cursor_fmt != 32) {

        avpriv_report_missing_feature(avctx, "Cursor format %d",

                                      cursor_fmt);

        return AVERROR_PATCHWELCOME;

    }



    tmp = av_realloc(c->cursor, cursor_stride * cursor_h);

    if (!tmp) {

        av_log(avctx, AV_LOG_ERROR, "Cannot allocate cursor buffer\n");

        return AVERROR(ENOMEM);

    }



    c->cursor        = tmp;

    c->cursor_w      = cursor_w;

    c->cursor_h      = cursor_h;

    c->cursor_hot_x  = cursor_hot_x;

    c->cursor_hot_y  = cursor_hot_y;

    c->cursor_fmt    = cursor_fmt;

    c->cursor_stride = cursor_stride;



    dst = c->cursor;

    switch (c->cursor_fmt) {

    case 1: // old monochrome

        for (j = 0; j < c->cursor_h; j++) {

            for (i = 0; i < c->cursor_w; i += 32) {

                bits = bytestream2_get_be32(gb);

                for (k = 0; k < 32; k++) {

                    dst[0] = !!(bits & 0x80000000);

                    dst += 4;

                    bits <<= 1;

                }

            }

        }



        dst = c->cursor;

        for (j = 0; j < c->cursor_h; j++) {

            for (i = 0; i < c->cursor_w; i += 32) {

                bits = bytestream2_get_be32(gb);

                for (k = 0; k < 32; k++) {

                    int mask_bit = !!(bits & 0x80000000);

                    switch (dst[0] * 2 + mask_bit) {

                    case 0:

                        dst[0] = 0xFF; dst[1] = 0x00;

                        dst[2] = 0x00; dst[3] = 0x00;

                        break;

                    case 1:

                        dst[0] = 0xFF; dst[1] = 0xFF;

                        dst[2] = 0xFF; dst[3] = 0xFF;

                        break;

                    default:

                        dst[0] = 0x00; dst[1] = 0x00;

                        dst[2] = 0x00; dst[3] = 0x00;

                    }

                    dst += 4;

                    bits <<= 1;

                }

            }

        }

        break;

    case 32: // full colour

        /* skip monochrome version of the cursor and decode RGBA instead */

        bytestream2_skip(gb, c->cursor_h * (FFALIGN(c->cursor_w, 32) >> 3));

        for (j = 0; j < c->cursor_h; j++) {

            for (i = 0; i < c->cursor_w; i++) {

                int val = bytestream2_get_be32(gb);

                *dst++ = val >>  0;

                *dst++ = val >>  8;

                *dst++ = val >> 16;

                *dst++ = val >> 24;

            }

        }

        break;

    default:

        return AVERROR_PATCHWELCOME;

    }

    return 0;

}
