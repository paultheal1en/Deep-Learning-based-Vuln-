av_cold void swri_resample_dsp_x86_init(ResampleContext *c)

{

    int av_unused mm_flags = av_get_cpu_flags();



    switch(c->format){

    case AV_SAMPLE_FMT_S16P:

        if (ARCH_X86_32 && EXTERNAL_MMXEXT(mm_flags)) {

            c->dsp.resample = c->linear ? ff_resample_linear_int16_mmxext

                                        : ff_resample_common_int16_mmxext;

        }

        if (EXTERNAL_SSE2(mm_flags)) {

            c->dsp.resample = c->linear ? ff_resample_linear_int16_sse2

                                        : ff_resample_common_int16_sse2;

        }

        if (EXTERNAL_XOP(mm_flags)) {

            c->dsp.resample = c->linear ? ff_resample_linear_int16_xop

                                        : ff_resample_common_int16_xop;

        }

        break;

    case AV_SAMPLE_FMT_FLTP:

        if (EXTERNAL_SSE(mm_flags)) {

            c->dsp.resample = c->linear ? ff_resample_linear_float_sse

                                        : ff_resample_common_float_sse;

        }

        if (EXTERNAL_AVX(mm_flags)) {

            c->dsp.resample = c->linear ? ff_resample_linear_float_avx

                                        : ff_resample_common_float_avx;

        }

        if (EXTERNAL_FMA3(mm_flags)) {

            c->dsp.resample = c->linear ? ff_resample_linear_float_fma3

                                        : ff_resample_common_float_fma3;

        }

        if (EXTERNAL_FMA4(mm_flags)) {

            c->dsp.resample = c->linear ? ff_resample_linear_float_fma4

                                        : ff_resample_common_float_fma4;

        }

        break;

    case AV_SAMPLE_FMT_DBLP:

        if (EXTERNAL_SSE2(mm_flags)) {

            c->dsp.resample = c->linear ? ff_resample_linear_double_sse2

                                        : ff_resample_common_double_sse2;

        }

        break;

    }

}
