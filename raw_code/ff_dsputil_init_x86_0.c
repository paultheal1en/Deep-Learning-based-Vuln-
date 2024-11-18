av_cold void ff_dsputil_init_x86(DSPContext *c, AVCodecContext *avctx)

{

    int cpu_flags = av_get_cpu_flags();



#if HAVE_7REGS && HAVE_INLINE_ASM

    if (cpu_flags & AV_CPU_FLAG_CMOV)

        c->add_hfyu_median_prediction = ff_add_hfyu_median_prediction_cmov;

#endif



    if (cpu_flags & AV_CPU_FLAG_MMX)

        dsputil_init_mmx(c, avctx, cpu_flags);



    if (cpu_flags & AV_CPU_FLAG_MMXEXT)

        dsputil_init_mmxext(c, avctx, cpu_flags);



    if (cpu_flags & AV_CPU_FLAG_SSE)

        dsputil_init_sse(c, avctx, cpu_flags);



    if (cpu_flags & AV_CPU_FLAG_SSE2)

        dsputil_init_sse2(c, avctx, cpu_flags);



    if (cpu_flags & AV_CPU_FLAG_SSSE3)

        dsputil_init_ssse3(c, avctx, cpu_flags);



    if (cpu_flags & AV_CPU_FLAG_SSE4)

        dsputil_init_sse4(c, avctx, cpu_flags);



    if (CONFIG_ENCODERS)

        ff_dsputilenc_init_mmx(c, avctx);

}
