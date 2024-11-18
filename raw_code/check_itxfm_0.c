static void check_itxfm(void)

{

    LOCAL_ALIGNED_32(uint8_t, src, [32 * 32 * 2]);

    LOCAL_ALIGNED_32(uint8_t, dst, [32 * 32 * 2]);

    LOCAL_ALIGNED_32(uint8_t, dst0, [32 * 32 * 2]);

    LOCAL_ALIGNED_32(uint8_t, dst1, [32 * 32 * 2]);

    LOCAL_ALIGNED_32(int16_t, coef, [32 * 32 * 2]);

    LOCAL_ALIGNED_32(int16_t, subcoef0, [32 * 32 * 2]);

    LOCAL_ALIGNED_32(int16_t, subcoef1, [32 * 32 * 2]);

    declare_func_emms(AV_CPU_FLAG_MMX | AV_CPU_FLAG_MMXEXT, void, uint8_t *dst, ptrdiff_t stride, int16_t *block, int eob);

    VP9DSPContext dsp;

    int y, x, tx, txtp, bit_depth, sub;

    static const char *const txtp_types[N_TXFM_TYPES] = {

        [DCT_DCT] = "dct_dct", [DCT_ADST] = "adst_dct",

        [ADST_DCT] = "dct_adst", [ADST_ADST] = "adst_adst"

    };



    for (bit_depth = 8; bit_depth <= 12; bit_depth += 2) {

        ff_vp9dsp_init(&dsp, bit_depth, 0);



        for (tx = TX_4X4; tx <= N_TXFM_SIZES /* 4 = lossless */; tx++) {

            int sz = 4 << (tx & 3);

            int n_txtps = tx < TX_32X32 ? N_TXFM_TYPES : 1;



            for (txtp = 0; txtp < n_txtps; txtp++) {

                if (check_func(dsp.itxfm_add[tx][txtp], "vp9_inv_%s_%dx%d_add_%d",

                               tx == 4 ? "wht_wht" : txtp_types[txtp], sz, sz,

                               bit_depth)) {

                    randomize_buffers();

                    ftx(coef, tx, txtp, sz, bit_depth);



                    for (sub = (txtp == 0) ? 1 : 2; sub <= sz; sub <<= 1) {

                        int eob;



                        if (sub < sz) {

                            eob = copy_subcoefs(subcoef0, coef, tx, txtp,

                                                sz, sub, bit_depth);

                        } else {

                            eob = sz * sz;

                            memcpy(subcoef0, coef, sz * sz * SIZEOF_COEF);

                        }



                        memcpy(dst0, dst, sz * sz * SIZEOF_PIXEL);

                        memcpy(dst1, dst, sz * sz * SIZEOF_PIXEL);

                        memcpy(subcoef1, subcoef0, sz * sz * SIZEOF_COEF);

                        call_ref(dst0, sz * SIZEOF_PIXEL, subcoef0, eob);

                        call_new(dst1, sz * SIZEOF_PIXEL, subcoef1, eob);

                        if (memcmp(dst0, dst1, sz * sz * SIZEOF_PIXEL) ||

                            !iszero(subcoef0, sz * sz * SIZEOF_COEF) ||

                            !iszero(subcoef1, sz * sz * SIZEOF_COEF))

                            fail();

                    }

                    bench_new(dst, sz * SIZEOF_PIXEL, coef, sz * sz);

                }

            }

        }

    }

    report("itxfm");

}
