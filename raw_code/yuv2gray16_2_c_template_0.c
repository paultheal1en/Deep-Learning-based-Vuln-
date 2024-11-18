yuv2gray16_2_c_template(SwsContext *c, const uint16_t *buf0,

                        const uint16_t *buf1, const uint16_t *ubuf0,

                        const uint16_t *ubuf1, const uint16_t *vbuf0,

                        const uint16_t *vbuf1, const uint16_t *abuf0,

                        const uint16_t *abuf1, uint8_t *dest, int dstW,

                        int yalpha, int uvalpha, int y,

                        enum PixelFormat target)

{

    int  yalpha1 = 4095 - yalpha; \

    int i;



    for (i = 0; i < (dstW >> 1); i++) {

        const int i2 = 2 * i;

        int Y1 = (buf0[i2  ] * yalpha1 + buf1[i2  ] * yalpha) >> 11;

        int Y2 = (buf0[i2+1] * yalpha1 + buf1[i2+1] * yalpha) >> 11;



        output_pixel(&dest[2 * i2 + 0], Y1);

        output_pixel(&dest[2 * i2 + 2], Y2);

    }

}
