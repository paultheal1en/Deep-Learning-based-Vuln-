void ff_hpeldsp_init_x86(HpelDSPContext *c, int flags)

{

    int cpu_flags = av_get_cpu_flags();



    if (INLINE_MMX(cpu_flags))

        hpeldsp_init_mmx(c, flags, cpu_flags);



    if (EXTERNAL_MMXEXT(cpu_flags))

        hpeldsp_init_mmxext(c, flags, cpu_flags);



    if (EXTERNAL_AMD3DNOW(cpu_flags))

        hpeldsp_init_3dnow(c, flags, cpu_flags);



    if (EXTERNAL_SSE2(cpu_flags))

        hpeldsp_init_sse2(c, flags, cpu_flags);

}
