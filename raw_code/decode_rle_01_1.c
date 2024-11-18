static int decode_rle(uint8_t *bitmap, int linesize, int w, int h,
                      const uint8_t *buf, int start, int buf_size, int is_8bit)
{
    GetBitContext gb;
    int bit_len;
    int x, y, len, color;
    uint8_t *d;
    if (start >= buf_size)
    bit_len = (buf_size - start) * 8;
    init_get_bits(&gb, buf + start, bit_len);
    x = 0;
    y = 0;
    d = bitmap;
    for(;;) {
        if (get_bits_count(&gb) > bit_len)
        if (is_8bit)
            len = decode_run_8bit(&gb, &color);
        else
            len = decode_run_2bit(&gb, &color);
        len = FFMIN(len, w - x);
        memset(d + x, color, len);
        x += len;
        if (x >= w) {
            y++;
            if (y >= h)
                break;
            d += linesize;
            x = 0;
            /* byte align */
            align_get_bits(&gb);
        }
    }
    return 0;
}