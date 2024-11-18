static void FUNC(transquant_bypass32x32)(uint8_t *_dst, int16_t *coeffs,

                                         ptrdiff_t stride)

{

    int x, y;

    pixel *dst = (pixel *)_dst;



    stride /= sizeof(pixel);



    for (y = 0; y < 32; y++) {

        for (x = 0; x < 32; x++) {

            dst[x] += *coeffs;

            coeffs++;

        }

        dst += stride;

    }

}
