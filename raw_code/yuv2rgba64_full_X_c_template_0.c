yuv2rgba64_full_X_c_template(SwsContext *c, const int16_t *lumFilter,

                       const int32_t **lumSrc, int lumFilterSize,

                       const int16_t *chrFilter, const int32_t **chrUSrc,

                       const int32_t **chrVSrc, int chrFilterSize,

                       const int32_t **alpSrc, uint16_t *dest, int dstW,

                       int y, enum AVPixelFormat target, int hasAlpha, int eightbytes)

{

    int i;

    int A = 0xffff<<14;



    for (i = 0; i < dstW; i++) {

        int j;

        int Y  = -0x40000000;

        int U  = -128 << 23; // 19

        int V  = -128 << 23;

        int R, G, B;



        for (j = 0; j < lumFilterSize; j++) {

            Y += lumSrc[j][i]  * (unsigned)lumFilter[j];

        }

        for (j = 0; j < chrFilterSize; j++) {;

            U += chrUSrc[j][i] * (unsigned)chrFilter[j];

            V += chrVSrc[j][i] * (unsigned)chrFilter[j];

        }



        if (hasAlpha) {

            A = -0x40000000;

            for (j = 0; j < lumFilterSize; j++) {

                A += alpSrc[j][i] * (unsigned)lumFilter[j];

            }

            A >>= 1;

            A += 0x20002000;

        }



        // 8bit: 12+15=27; 16-bit: 12+19=31

        Y  >>= 14; // 10

        Y += 0x10000;

        U  >>= 14;

        V  >>= 14;



        // 8bit: 27 -> 17bit, 16bit: 31 - 14 = 17bit

        Y -= c->yuv2rgb_y_offset;

        Y *= c->yuv2rgb_y_coeff;

        Y += 1 << 13; // 21

        // 8bit: 17 + 13bit = 30bit, 16bit: 17 + 13bit = 30bit



        R = V * c->yuv2rgb_v2r_coeff;

        G = V * c->yuv2rgb_v2g_coeff + U * c->yuv2rgb_u2g_coeff;

        B =                            U * c->yuv2rgb_u2b_coeff;



        // 8bit: 30 - 22 = 8bit, 16bit: 30bit - 14 = 16bit

        output_pixel(&dest[0], av_clip_uintp2(R_B + Y, 30) >> 14);

        output_pixel(&dest[1], av_clip_uintp2(  G + Y, 30) >> 14);

        output_pixel(&dest[2], av_clip_uintp2(B_R + Y, 30) >> 14);

        if (eightbytes) {

            output_pixel(&dest[3], av_clip_uintp2(A, 30) >> 14);

            dest += 4;

        } else {

            dest += 3;

        }

    }

}
