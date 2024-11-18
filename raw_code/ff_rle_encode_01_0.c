int ff_rle_encode(uint8_t *outbuf, int out_size, const uint8_t *ptr , int bpp, int w,

                  int add_rep, int xor_rep, int add_raw, int xor_raw)

{

    int count, x;

    uint8_t *out = outbuf;



    for(x = 0; x < w; x += count) {

        /* see if we can encode the next set of pixels with RLE */

        if((count = count_pixels(ptr, w-x, bpp, 1)) > 1) {

            if(out + bpp + 1 > outbuf + out_size) return -1;

            *out++ = (count ^ xor_rep) + add_rep;

            memcpy(out, ptr, bpp);

            out += bpp;

        } else {

            /* fall back on uncompressed */

            count = count_pixels(ptr, w-x, bpp, 0);

            *out++ = (count ^ xor_raw) + add_raw;



            if(out + bpp*count > outbuf + out_size) return -1;

            memcpy(out, ptr, bpp * count);

            out += bpp * count;

        }



        ptr += count * bpp;

    }



    return out - outbuf;

}
