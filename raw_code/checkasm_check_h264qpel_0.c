void checkasm_check_h264qpel(void)

{

    LOCAL_ALIGNED_16(uint8_t, buf0, [BUF_SIZE]);

    LOCAL_ALIGNED_16(uint8_t, buf1, [BUF_SIZE]);

    LOCAL_ALIGNED_16(uint8_t, dst0, [BUF_SIZE]);

    LOCAL_ALIGNED_16(uint8_t, dst1, [BUF_SIZE]);

    H264QpelContext h;

    int op, bit_depth, i, j;



    for (op = 0; op < 2; op++) {

        qpel_mc_func (*tab)[16] = op ? h.avg_h264_qpel_pixels_tab : h.put_h264_qpel_pixels_tab;

        const char *op_name = op ? "avg" : "put";



        for (bit_depth = 8; bit_depth <= 10; bit_depth++) {

            ff_h264qpel_init(&h, bit_depth);

            for (i = 0; i < (op ? 3 : 4); i++) {

                int size = 16 >> i;

                for (j = 0; j < 16; j++)

                    if (check_func(tab[i][j], "%s_h264_qpel_%d_mc%d%d_%d", op_name, size, j & 3, j >> 2, bit_depth)) {

                        randomize_buffers();

                        call_ref(dst0, src0, (ptrdiff_t)size * SIZEOF_PIXEL);

                        call_new(dst1, src1, (ptrdiff_t)size * SIZEOF_PIXEL);

                        if (memcmp(buf0, buf1, BUF_SIZE) || memcmp(dst0, dst1, BUF_SIZE))

                            fail();

                        bench_new(dst1, src1, (ptrdiff_t)size * SIZEOF_PIXEL);

                    }

            }

        }

        report("%s", op_name);

    }

}
