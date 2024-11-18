av_cold void ff_vp56dsp_init_x86(VP56DSPContext* c, enum AVCodecID codec)

{

#if HAVE_YASM

    int mm_flags = av_get_cpu_flags();



    if (CONFIG_VP6_DECODER && codec == AV_CODEC_ID_VP6) {

#if ARCH_X86_32

        if (mm_flags & AV_CPU_FLAG_MMX) {

            c->vp6_filter_diag4 = ff_vp6_filter_diag4_mmx;

        }

#endif



        if (mm_flags & AV_CPU_FLAG_SSE2) {

            c->vp6_filter_diag4 = ff_vp6_filter_diag4_sse2;

        }

    }

#endif

}
