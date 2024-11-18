av_cold void ff_vc1dsp_init_x86(VC1DSPContext *dsp)

{

    int cpu_flags = av_get_cpu_flags();



    if (INLINE_MMX(cpu_flags))

        ff_vc1dsp_init_mmx(dsp);



    if (INLINE_MMXEXT(cpu_flags))

        ff_vc1dsp_init_mmxext(dsp);



#define ASSIGN_LF(EXT) \

        dsp->vc1_v_loop_filter4  = ff_vc1_v_loop_filter4_ ## EXT; \

        dsp->vc1_h_loop_filter4  = ff_vc1_h_loop_filter4_ ## EXT; \

        dsp->vc1_v_loop_filter8  = ff_vc1_v_loop_filter8_ ## EXT; \

        dsp->vc1_h_loop_filter8  = ff_vc1_h_loop_filter8_ ## EXT; \

        dsp->vc1_v_loop_filter16 = vc1_v_loop_filter16_ ## EXT; \

        dsp->vc1_h_loop_filter16 = vc1_h_loop_filter16_ ## EXT



#if HAVE_YASM

    if (cpu_flags & AV_CPU_FLAG_MMX) {

        dsp->put_no_rnd_vc1_chroma_pixels_tab[0] = ff_put_vc1_chroma_mc8_nornd_mmx;

    }



    if (cpu_flags & AV_CPU_FLAG_MMXEXT) {

        ASSIGN_LF(mmxext);

        dsp->avg_no_rnd_vc1_chroma_pixels_tab[0] = ff_avg_vc1_chroma_mc8_nornd_mmxext;



        dsp->avg_vc1_mspel_pixels_tab[0]         = avg_vc1_mspel_mc00_mmxext;

    } else if (cpu_flags & AV_CPU_FLAG_3DNOW) {

        dsp->avg_no_rnd_vc1_chroma_pixels_tab[0] = ff_avg_vc1_chroma_mc8_nornd_3dnow;

    }



    if (cpu_flags & AV_CPU_FLAG_SSE2) {

        dsp->vc1_v_loop_filter8  = ff_vc1_v_loop_filter8_sse2;

        dsp->vc1_h_loop_filter8  = ff_vc1_h_loop_filter8_sse2;

        dsp->vc1_v_loop_filter16 = vc1_v_loop_filter16_sse2;

        dsp->vc1_h_loop_filter16 = vc1_h_loop_filter16_sse2;

    }

    if (cpu_flags & AV_CPU_FLAG_SSSE3) {

        ASSIGN_LF(ssse3);

        dsp->put_no_rnd_vc1_chroma_pixels_tab[0] = ff_put_vc1_chroma_mc8_nornd_ssse3;

        dsp->avg_no_rnd_vc1_chroma_pixels_tab[0] = ff_avg_vc1_chroma_mc8_nornd_ssse3;

    }

    if (cpu_flags & AV_CPU_FLAG_SSE4) {

        dsp->vc1_h_loop_filter8  = ff_vc1_h_loop_filter8_sse4;

        dsp->vc1_h_loop_filter16 = vc1_h_loop_filter16_sse4;

    }

#endif /* HAVE_YASM */

}
