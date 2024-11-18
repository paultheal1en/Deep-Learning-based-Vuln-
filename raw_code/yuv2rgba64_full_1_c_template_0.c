yuv2rgba64_full_1_c_template(SwsContext *c, const int32_t *buf0,

                       const int32_t *ubuf[2], const int32_t *vbuf[2],

                       const int32_t *abuf0, uint16_t *dest, int dstW,

                       int uvalpha, int y, enum AVPixelFormat target, int hasAlpha, int eightbytes)

{

    const int32_t *ubuf0 = ubuf[0], *vbuf0 = vbuf[0];

    int i;

    int A = 0xffff<<14;



    if (uvalpha < 2048) {

        for (i = 0; i < dstW; i++) {

            int Y  = (buf0[i]) >> 2;

            int U  = (ubuf0[i] + (-128 << 11)) >> 2;

            int V  = (vbuf0[i] + (-128 << 11)) >> 2;

            int R, G, B;



            Y -= c->yuv2rgb_y_offset;

            Y *= c->yuv2rgb_y_coeff;

            Y += 1 << 13;



            if (hasAlpha) {

                A = abuf0[i] << 11;



                A += 1 << 13;

            }



            R = V * c->yuv2rgb_v2r_coeff;

            G = V * c->yuv2rgb_v2g_coeff + U * c->yuv2rgb_u2g_coeff;

            B =                            U * c->yuv2rgb_u2b_coeff;



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

    } else {

        const int32_t *ubuf1 = ubuf[1], *vbuf1 = vbuf[1];

        int A = 0xffff<<14;

        for (i = 0; i < dstW; i++) {

            int Y  = (buf0[i]    ) >> 2;

            int U  = (ubuf0[i] + ubuf1[i] + (-128 << 12)) >> 3;

            int V  = (vbuf0[i] + vbuf1[i] + (-128 << 12)) >> 3;

            int R, G, B;



            Y -= c->yuv2rgb_y_offset;

            Y *= c->yuv2rgb_y_coeff;

            Y += 1 << 13;



            if (hasAlpha) {

                A = abuf0[i] << 11;



                A += 1 << 13;

            }



            R = V * c->yuv2rgb_v2r_coeff;

            G = V * c->yuv2rgb_v2g_coeff + U * c->yuv2rgb_u2g_coeff;

            B =                            U * c->yuv2rgb_u2b_coeff;



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

}
