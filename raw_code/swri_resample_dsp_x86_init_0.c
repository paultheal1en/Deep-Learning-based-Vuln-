void swri_resample_dsp_x86_init(ResampleContext *c)

{

    int av_unused mm_flags = av_get_cpu_flags();



    switch(c->format){

    case AV_SAMPLE_FMT_S16P:

        if (ARCH_X86_32 && HAVE_MMXEXT_EXTERNAL && mm_flags & AV_CPU_FLAG_MMX2) {

            c->dsp.resample = c->linear ? ff_resample_linear_int16_mmxext

                                        : ff_resample_common_int16_mmxext;

        }

        if (HAVE_SSE2_EXTERNAL && mm_flags & AV_CPU_FLAG_SSE2) {

            c->dsp.resample = c->linear ? ff_resample_linear_int16_sse2

                                        : ff_resample_common_int16_sse2;

        }

        if (HAVE_XOP_EXTERNAL && mm_flags & AV_CPU_FLAG_XOP) {

            c->dsp.resample = c->linear ? ff_resample_linear_int16_xop

                                        : ff_resample_common_int16_xop;

        }

        break;

    case AV_SAMPLE_FMT_FLTP:

        if (HAVE_SSE_EXTERNAL && mm_flags & AV_CPU_FLAG_SSE) {

            c->dsp.resample = c->linear ? ff_resample_linear_float_sse

                                        : ff_resample_common_float_sse;

        }

        if (HAVE_AVX_EXTERNAL && mm_flags & AV_CPU_FLAG_AVX) {

            c->dsp.resample = c->linear ? ff_resample_linear_float_avx

                                        : ff_resample_common_float_avx;

        }

        if (HAVE_FMA3_EXTERNAL && mm_flags & AV_CPU_FLAG_FMA3) {

            c->dsp.resample = c->linear ? ff_resample_linear_float_fma3

                                        : ff_resample_common_float_fma3;

        }

        if (HAVE_FMA4_EXTERNAL && mm_flags & AV_CPU_FLAG_FMA4) {

            c->dsp.resample = c->linear ? ff_resample_linear_float_fma4

                                        : ff_resample_common_float_fma4;

        }

        break;

    case AV_SAMPLE_FMT_DBLP:

        if (HAVE_SSE2_EXTERNAL && mm_flags & AV_CPU_FLAG_SSE2) {

            c->dsp.resample = c->linear ? ff_resample_linear_double_sse2

                                        : ff_resample_common_double_sse2;

        }

        break;

    }

}
