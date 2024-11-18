void ff_limiter_init_x86(LimiterDSPContext *dsp, int bpp)

{

    int cpu_flags = av_get_cpu_flags();



    if (ARCH_X86_64 && EXTERNAL_SSE2(cpu_flags)) {

        if (bpp <= 8) {

            dsp->limiter = ff_limiter_8bit_sse2;

        }

    }

    if (ARCH_X86_64 && EXTERNAL_SSE4(cpu_flags)) {

        if (bpp > 8) {

            dsp->limiter = ff_limiter_16bit_sse4;

        }

    }

}
