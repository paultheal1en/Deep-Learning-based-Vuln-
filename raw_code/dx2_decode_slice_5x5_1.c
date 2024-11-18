static int dx2_decode_slice_5x5(GetBitContext *gb, AVFrame *frame,

                                int line, int left, uint8_t lru[3][8],

                                int is_565)

{

    int x, y;

    int r, g, b;

    int width    = frame->width;

    int stride   = frame->linesize[0];

    uint8_t *dst = frame->data[0] + stride * line;



    for (y = 0; y < left && get_bits_left(gb) > 16; y++) {

        for (x = 0; x < width; x++) {

            b = decode_sym_565(gb, lru[0], 5);

            g = decode_sym_565(gb, lru[1], is_565 ? 6 : 5);

            r = decode_sym_565(gb, lru[2], 5);

            dst[x * 3 + 0] = (r << 3) | (r >> 2);

            dst[x * 3 + 1] = is_565 ? (g << 2) | (g >> 4) : (g << 3) | (g >> 2);

            dst[x * 3 + 2] = (b << 3) | (b >> 2);

        }



        dst += stride;

    }



    return y;

}
