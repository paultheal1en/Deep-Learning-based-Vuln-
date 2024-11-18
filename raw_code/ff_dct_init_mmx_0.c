av_cold void ff_dct_init_mmx(DCTContext *s)

{

#if HAVE_YASM

    int has_vectors = av_get_cpu_flags();

    if (has_vectors & AV_CPU_FLAG_SSE && HAVE_SSE)

        s->dct32 = ff_dct32_float_sse;

    if (has_vectors & AV_CPU_FLAG_SSE2 && HAVE_SSE)

        s->dct32 = ff_dct32_float_sse2;

    if (has_vectors & AV_CPU_FLAG_AVX && HAVE_AVX)

        s->dct32 = ff_dct32_float_avx;

#endif

}
