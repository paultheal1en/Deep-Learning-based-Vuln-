av_cold void ff_yadif_init_x86(YADIFContext *yadif)

{

    int cpu_flags = av_get_cpu_flags();



    if (HAVE_MMX && cpu_flags & AV_CPU_FLAG_MMX)

        yadif->filter_line = yadif_filter_line_mmx;

    if (HAVE_SSE && cpu_flags & AV_CPU_FLAG_SSE2)

        yadif->filter_line = yadif_filter_line_sse2;

    if (HAVE_SSSE3 && cpu_flags & AV_CPU_FLAG_SSSE3)

        yadif->filter_line = yadif_filter_line_ssse3;

}
