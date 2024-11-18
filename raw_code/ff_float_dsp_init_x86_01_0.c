av_cold void ff_float_dsp_init_x86(AVFloatDSPContext *fdsp)

{

    int cpu_flags = av_get_cpu_flags();



#if HAVE_6REGS && HAVE_INLINE_ASM

    if (INLINE_AMD3DNOWEXT(cpu_flags)) {

        fdsp->vector_fmul_window  = vector_fmul_window_3dnowext;

    }

    if (INLINE_SSE(cpu_flags)) {

        fdsp->vector_fmul_window = vector_fmul_window_sse;

    }

#endif

    if (EXTERNAL_SSE(cpu_flags)) {

        fdsp->vector_fmul = ff_vector_fmul_sse;

        fdsp->vector_fmac_scalar = ff_vector_fmac_scalar_sse;

        fdsp->vector_fmul_scalar = ff_vector_fmul_scalar_sse;

        fdsp->vector_fmul_add    = ff_vector_fmul_add_sse;

        fdsp->vector_fmul_reverse = ff_vector_fmul_reverse_sse;

        fdsp->scalarproduct_float = ff_scalarproduct_float_sse;

        fdsp->butterflies_float   = ff_butterflies_float_sse;

    }

    if (EXTERNAL_SSE2(cpu_flags)) {

        fdsp->vector_dmul_scalar = ff_vector_dmul_scalar_sse2;

    }

    if (EXTERNAL_AVX(cpu_flags)) {

        fdsp->vector_fmul = ff_vector_fmul_avx;

        fdsp->vector_fmac_scalar = ff_vector_fmac_scalar_avx;

        fdsp->vector_dmul_scalar = ff_vector_dmul_scalar_avx;

        fdsp->vector_fmul_add    = ff_vector_fmul_add_avx;

        fdsp->vector_fmul_reverse = ff_vector_fmul_reverse_avx;

    }

}
