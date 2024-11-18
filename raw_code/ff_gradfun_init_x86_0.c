av_cold void ff_gradfun_init_x86(GradFunContext *gf)

{

    int cpu_flags = av_get_cpu_flags();



    if (HAVE_MMX2 && cpu_flags & AV_CPU_FLAG_MMX2)

        gf->filter_line = gradfun_filter_line_mmx2;

    if (HAVE_SSSE3 && cpu_flags & AV_CPU_FLAG_SSSE3)

        gf->filter_line = gradfun_filter_line_ssse3;

    if (HAVE_SSE && cpu_flags & AV_CPU_FLAG_SSE2)

        gf->blur_line = gradfun_blur_line_sse2;

}
