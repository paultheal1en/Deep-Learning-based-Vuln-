void ff_proresdsp_x86_init(ProresDSPContext *dsp)

{

#if ARCH_X86_64 && HAVE_YASM

    int flags = av_get_cpu_flags();



    if (flags & AV_CPU_FLAG_SSE2) {

        dsp->idct_permutation_type = FF_TRANSPOSE_IDCT_PERM;

        dsp->idct_put = ff_prores_idct_put_10_sse2;

    }



    if (flags & AV_CPU_FLAG_SSE4) {

        dsp->idct_permutation_type = FF_TRANSPOSE_IDCT_PERM;

        dsp->idct_put = ff_prores_idct_put_10_sse4;

    }



#if HAVE_AVX

    if (flags & AV_CPU_FLAG_AVX) {

        dsp->idct_permutation_type = FF_TRANSPOSE_IDCT_PERM;

        dsp->idct_put = ff_prores_idct_put_10_avx;

    }

#endif /* HAVE_AVX */

#endif /* ARCH_X86_64 && HAVE_YASM */

}
