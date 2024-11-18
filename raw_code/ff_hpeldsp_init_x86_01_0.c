av_cold void ff_hpeldsp_init_x86(HpelDSPContext *c, int flags)

{

    int cpu_flags = av_get_cpu_flags();



    if (INLINE_MMX(cpu_flags))

        hpeldsp_init_mmx(c, flags);



    if (EXTERNAL_AMD3DNOW(cpu_flags))

        hpeldsp_init_3dnow(c, flags);



    if (EXTERNAL_MMXEXT(cpu_flags))

        hpeldsp_init_mmxext(c, flags);



    if (EXTERNAL_SSE2_FAST(cpu_flags))

        hpeldsp_init_sse2_fast(c, flags);



    if (CONFIG_VP3_DECODER)

        ff_hpeldsp_vp3_init_x86(c, cpu_flags, flags);

}
