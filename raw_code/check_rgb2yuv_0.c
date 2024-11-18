static void check_rgb2yuv(void)

{

    declare_func(void, uint8_t *dst[3], ptrdiff_t dst_stride[3],

                 int16_t *src[3], ptrdiff_t src_stride,

                 int w, int h, const int16_t coeff[3][3][8],

                 const int16_t off[8]);

    ColorSpaceDSPContext dsp;

    int odepth, fmt, n;

    LOCAL_ALIGNED_32(int16_t, src_y, [W * H * 2]);

    LOCAL_ALIGNED_32(int16_t, src_u, [W * H * 2]);

    LOCAL_ALIGNED_32(int16_t, src_v, [W * H * 2]);

    int16_t *src[3] = { src_y, src_u, src_v };

    LOCAL_ALIGNED_32(uint8_t, dst0_y, [W * H]);

    LOCAL_ALIGNED_32(uint8_t, dst0_u, [W * H]);

    LOCAL_ALIGNED_32(uint8_t, dst0_v, [W * H]);

    LOCAL_ALIGNED_32(uint8_t, dst1_y, [W * H]);

    LOCAL_ALIGNED_32(uint8_t, dst1_u, [W * H]);

    LOCAL_ALIGNED_32(uint8_t, dst1_v, [W * H]);

    uint8_t *dst0[3] = { dst0_y, dst0_u, dst0_v }, *dst1[3] = { dst1_y, dst1_u, dst1_v };

    LOCAL_ALIGNED_32(int16_t, offset, [8]);

    LOCAL_ALIGNED_32(int16_t, coeff_buf, [3 * 3 * 8]);

    int16_t (*coeff)[3][8] = (int16_t(*)[3][8]) coeff_buf;



    ff_colorspacedsp_init(&dsp);

    for (n = 0; n < 8; n++) {

        offset[n] = 16;



        // these somewhat resemble bt601/smpte170m coefficients

        coeff[0][0][n] = lrint(0.3 * (1 << 14));

        coeff[0][1][n] = lrint(0.6 * (1 << 14));

        coeff[0][2][n] = lrint(0.1 * (1 << 14));

        coeff[1][0][n] = lrint(-0.15 * (1 << 14));

        coeff[1][1][n] = lrint(-0.35 * (1 << 14));

        coeff[1][2][n] = lrint(0.5 * (1 << 14));

        coeff[2][0][n] = lrint(0.5 * (1 << 14));

        coeff[2][1][n] = lrint(-0.42 * (1 << 14));

        coeff[2][2][n] = lrint(-0.08 * (1 << 14));

    }

    for (odepth = 0; odepth < 3; odepth++) {

        for (fmt = 0; fmt < 3; fmt++) {

            if (check_func(dsp.rgb2yuv[odepth][fmt],

                           "ff_colorspacedsp_rgb2yuv_%sp%d",

                           format_string[fmt], odepth * 2 + 8)) {

                int ss_w = !!fmt, ss_h = fmt == 2;

                int y_dst_stride = W << !!odepth;

                int uv_dst_stride = y_dst_stride >> ss_w;



                randomize_buffers();

                call_ref(dst0, (ptrdiff_t[3]) { y_dst_stride, uv_dst_stride, uv_dst_stride },

                         src, W, W, H, coeff, offset);

                call_new(dst1, (ptrdiff_t[3]) { y_dst_stride, uv_dst_stride, uv_dst_stride },

                         src, W, W, H, coeff, offset);

                if (memcmp(dst0[0], dst1[0], H * y_dst_stride) ||

                    memcmp(dst0[1], dst1[1], H * uv_dst_stride >> ss_h) ||

                    memcmp(dst0[2], dst1[2], H * uv_dst_stride >> ss_h)) {

                    fail();

                }

            }

        }

    }



    report("rgb2yuv");

}
