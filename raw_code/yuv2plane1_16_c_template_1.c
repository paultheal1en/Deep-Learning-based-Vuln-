yuv2plane1_16_c_template(const int32_t *src, uint16_t *dest, int dstW,

                         int big_endian, int output_bits)

{

    int i;

    int shift = 19 - output_bits;



    for (i = 0; i < dstW; i++) {

        int val = src[i] + (1 << (shift - 1));

        output_pixel(&dest[i], val, 0, uint);

    }

}
