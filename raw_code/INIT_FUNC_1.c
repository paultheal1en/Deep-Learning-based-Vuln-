av_cold void INIT_FUNC(VP9DSPContext *dsp, int bitexact)

{

#if HAVE_YASM

    int cpu_flags = av_get_cpu_flags();



#define init_lpf_8_func(idx1, idx2, dir, wd, bpp, opt) \

    dsp->loop_filter_8[idx1][idx2] = ff_vp9_loop_filter_##dir##_##wd##_##bpp##_##opt

#define init_lpf_16_func(idx, dir, bpp, opt) \

    dsp->loop_filter_16[idx] = loop_filter_##dir##_16_##bpp##_##opt

#define init_lpf_mix2_func(idx1, idx2, idx3, dir, wd1, wd2, bpp, opt) \

    dsp->loop_filter_mix2[idx1][idx2][idx3] = loop_filter_##dir##_##wd1##wd2##_##bpp##_##opt



#define init_lpf_funcs(bpp, opt) \

    init_lpf_8_func(0, 0, h,  4, bpp, opt); \

    init_lpf_8_func(0, 1, v,  4, bpp, opt); \

    init_lpf_8_func(1, 0, h,  8, bpp, opt); \

    init_lpf_8_func(1, 1, v,  8, bpp, opt); \

    init_lpf_8_func(2, 0, h, 16, bpp, opt); \

    init_lpf_8_func(2, 1, v, 16, bpp, opt); \

    init_lpf_16_func(0, h, bpp, opt); \

    init_lpf_16_func(1, v, bpp, opt); \

    init_lpf_mix2_func(0, 0, 0, h, 4, 4, bpp, opt); \

    init_lpf_mix2_func(0, 1, 0, h, 4, 8, bpp, opt); \

    init_lpf_mix2_func(1, 0, 0, h, 8, 4, bpp, opt); \

    init_lpf_mix2_func(1, 1, 0, h, 8, 8, bpp, opt); \

    init_lpf_mix2_func(0, 0, 1, v, 4, 4, bpp, opt); \

    init_lpf_mix2_func(0, 1, 1, v, 4, 8, bpp, opt); \

    init_lpf_mix2_func(1, 0, 1, v, 8, 4, bpp, opt); \

    init_lpf_mix2_func(1, 1, 1, v, 8, 8, bpp, opt)



#define init_itx_func(idxa, idxb, typea, typeb, size, bpp, opt) \

    dsp->itxfm_add[idxa][idxb] = \

        ff_vp9_##typea##_##typeb##_##size##x##size##_add_##bpp##_##opt;

#define init_itx_func_one(idx, typea, typeb, size, bpp, opt) \

    init_itx_func(idx, DCT_DCT,   typea, typeb, size, bpp, opt); \

    init_itx_func(idx, ADST_DCT,  typea, typeb, size, bpp, opt); \

    init_itx_func(idx, DCT_ADST,  typea, typeb, size, bpp, opt); \

    init_itx_func(idx, ADST_ADST, typea, typeb, size, bpp, opt)

#define init_itx_funcs(idx, size, bpp, opt) \

    init_itx_func(idx, DCT_DCT,   idct,  idct,  size, bpp, opt); \

    init_itx_func(idx, ADST_DCT,  idct,  iadst, size, bpp, opt); \

    init_itx_func(idx, DCT_ADST,  iadst, idct,  size, bpp, opt); \

    init_itx_func(idx, ADST_ADST, iadst, iadst, size, bpp, opt); \



    if (EXTERNAL_MMXEXT(cpu_flags)) {

        init_ipred_func(tm, TM_VP8, 4, BPC, mmxext);

        if (!bitexact) {

            init_itx_func_one(4 /* lossless */, iwht, iwht, 4, BPC, mmxext);

#if BPC == 10

            init_itx_func(TX_4X4, DCT_DCT, idct, idct, 4, 10, mmxext);

#endif

        }

    }



    if (EXTERNAL_SSE2(cpu_flags)) {

        init_subpel3(0, put, BPC, sse2);

        init_subpel3(1, avg, BPC, sse2);

        init_lpf_funcs(BPC, sse2);

        init_8_16_32_ipred_funcs(tm, TM_VP8, BPC, sse2);

#if BPC == 10

        if (!bitexact) {

            init_itx_func(TX_4X4, ADST_DCT,  idct,  iadst, 4, 10, sse2);

            init_itx_func(TX_4X4, DCT_ADST,  iadst, idct,  4, 10, sse2);

            init_itx_func(TX_4X4, ADST_ADST, iadst, iadst, 4, 10, sse2);

        }



#endif

    }



    if (EXTERNAL_SSSE3(cpu_flags)) {

        init_lpf_funcs(BPC, ssse3);

#if BPC == 10

        if (!bitexact) {

            init_itx_funcs(TX_4X4, 4, BPC, ssse3);

        }

#endif

    }



    if (EXTERNAL_AVX(cpu_flags)) {

        init_lpf_funcs(BPC, avx);

    }



    if (EXTERNAL_AVX2(cpu_flags)) {

#if HAVE_AVX2_EXTERNAL

        init_subpel3_32_64(0,  put, BPC, avx2);

        init_subpel3_32_64(1,  avg, BPC, avx2);

        init_subpel2(2, 0, 16, put, BPC, avx2);

        init_subpel2(2, 1, 16, avg, BPC, avx2);

#endif

    }



#endif /* HAVE_YASM */



    ff_vp9dsp_init_16bpp_x86(dsp);

}