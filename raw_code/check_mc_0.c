static void check_mc(void)

{

    LOCAL_ALIGNED_32(uint8_t, buf, [72 * 72 * 2]);

    LOCAL_ALIGNED_32(uint8_t, dst0, [64 * 64 * 2]);

    LOCAL_ALIGNED_32(uint8_t, dst1, [64 * 64 * 2]);

    VP9DSPContext dsp;

    int op, hsize, bit_depth, filter, dx, dy;

    declare_func(void, uint8_t *dst, ptrdiff_t dst_stride,

                 const uint8_t *ref, ptrdiff_t ref_stride,

                 int h, int mx, int my);

    static const char *const filter_names[4] = {

        "8tap_smooth", "8tap_regular", "8tap_sharp", "bilin"

    };

    static const char *const subpel_names[2][2] = { { "", "h" }, { "v", "hv" } };

    static const char *const op_names[2] = { "put", "avg" };

    char str[256];



    for (op = 0; op < 2; op++) {

        for (bit_depth = 8; bit_depth <= 12; bit_depth += 2) {

            ff_vp9dsp_init(&dsp, bit_depth, 0);

            for (hsize = 0; hsize < 5; hsize++) {

                int size = 64 >> hsize;



                for (filter = 0; filter < 4; filter++) {

                    for (dx = 0; dx < 2; dx++) {

                        for (dy = 0; dy < 2; dy++) {

                            if (dx || dy) {

                                sprintf(str, "%s_%s_%d%s", op_names[op],

                                        filter_names[filter], size,

                                        subpel_names[dy][dx]);

                            } else {

                                sprintf(str, "%s%d", op_names[op], size);

                            }

                            if (check_func(dsp.mc[hsize][filter][op][dx][dy],

                                           "vp9_%s_%dbpp", str, bit_depth)) {

                                int mx = dx ? 1 + (rnd() % 14) : 0;

                                int my = dy ? 1 + (rnd() % 14) : 0;

                                randomize_buffers();

                                call_ref(dst0, size * SIZEOF_PIXEL,

                                         src, SRC_BUF_STRIDE * SIZEOF_PIXEL,

                                         size, mx, my);

                                call_new(dst1, size * SIZEOF_PIXEL,

                                         src, SRC_BUF_STRIDE * SIZEOF_PIXEL,

                                         size, mx, my);

                                if (memcmp(dst0, dst1, DST_BUF_SIZE))

                                    fail();



                                // simd implementations for each filter of subpel

                                // functions are identical

                                if (filter >= 1 && filter <= 2) continue;

                                // 10/12 bpp for bilin are identical

                                if (bit_depth == 12 && filter == 3) continue;



                                bench_new(dst1, size * SIZEOF_PIXEL,

                                          src, SRC_BUF_STRIDE * SIZEOF_PIXEL,

                                          size, mx, my);

                            }

                        }

                    }

                }

            }

        }

    }

    report("mc");

}
