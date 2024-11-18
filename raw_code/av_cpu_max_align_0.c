size_t av_cpu_max_align(void)

{

    int flags = av_get_cpu_flags();



    if (flags & AV_CPU_FLAG_AVX)

        return 32;

    if (flags & (AV_CPU_FLAG_ALTIVEC | AV_CPU_FLAG_SSE | AV_CPU_FLAG_NEON))

        return 16;



    return 8;

}
