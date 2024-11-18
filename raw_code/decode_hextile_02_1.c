static int decode_hextile(VmncContext *c, uint8_t* dst, GetByteContext *gb,

                          int w, int h, int stride)

{

    int i, j, k;

    int bg = 0, fg = 0, rects, color, flags, xy, wh;

    const int bpp = c->bpp2;

    uint8_t *dst2;

    int bw = 16, bh = 16;



    for (j = 0; j < h; j += 16) {

        dst2 = dst;

        bw   = 16;

        if (j + 16 > h)

            bh = h - j;

        for (i = 0; i < w; i += 16, dst2 += 16 * bpp) {

            if (bytestream2_get_bytes_left(gb) <= 0) {

                av_log(c->avctx, AV_LOG_ERROR, "Premature end of data!\n");

                return AVERROR_INVALIDDATA;

            }

            if (i + 16 > w)

                bw = w - i;

            flags = bytestream2_get_byte(gb);

            if (flags & HT_RAW) {

                if (bytestream2_get_bytes_left(gb) < bw * bh * bpp) {

                    av_log(c->avctx, AV_LOG_ERROR, "Premature end of data!\n");

                    return AVERROR_INVALIDDATA;

                }

                paint_raw(dst2, bw, bh, gb, bpp, c->bigendian, stride);

            } else {

                if (flags & HT_BKG)

                    bg = vmnc_get_pixel(gb, bpp, c->bigendian);

                if (flags & HT_FG)

                    fg = vmnc_get_pixel(gb, bpp, c->bigendian);

                rects = 0;

                if (flags & HT_SUB)

                    rects = bytestream2_get_byte(gb);

                color = !!(flags & HT_CLR);



                paint_rect(dst2, 0, 0, bw, bh, bg, bpp, stride);



                if (bytestream2_get_bytes_left(gb) < rects * (color * bpp + 2)) {

                    av_log(c->avctx, AV_LOG_ERROR, "Premature end of data!\n");

                    return AVERROR_INVALIDDATA;

                }

                for (k = 0; k < rects; k++) {

                    if (color)

                        fg = vmnc_get_pixel(gb, bpp, c->bigendian);

                    xy = bytestream2_get_byte(gb);

                    wh = bytestream2_get_byte(gb);

                    paint_rect(dst2, xy >> 4, xy & 0xF,

                               (wh>>4)+1, (wh & 0xF)+1, fg, bpp, stride);

                }

            }

        }

        dst += stride * 16;

    }

    return 0;

}
