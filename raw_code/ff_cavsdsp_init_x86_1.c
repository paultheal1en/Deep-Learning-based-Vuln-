av_cold void ff_cavsdsp_init_x86(CAVSDSPContext *c, AVCodecContext *avctx)

{

    av_unused int cpu_flags = av_get_cpu_flags();



    cavsdsp_init_mmx(c, avctx);

#if HAVE_AMD3DNOW_INLINE

    if (INLINE_AMD3DNOW(cpu_flags))

        cavsdsp_init_3dnow(c, avctx);

#endif /* HAVE_AMD3DNOW_INLINE */

#if HAVE_MMXEXT_INLINE

    if (INLINE_MMXEXT(cpu_flags)) {

        DSPFUNC(put, 0, 16, mmxext);

        DSPFUNC(put, 1,  8, mmxext);

        DSPFUNC(avg, 0, 16, mmxext);

        DSPFUNC(avg, 1,  8, mmxext);

    }

#endif

#if HAVE_MMX_EXTERNAL

    if (EXTERNAL_MMXEXT(cpu_flags)) {

        c->avg_cavs_qpel_pixels_tab[0][0] = avg_cavs_qpel16_mc00_mmxext;

        c->avg_cavs_qpel_pixels_tab[1][0] = avg_cavs_qpel8_mc00_mmxext;

    }

#endif

#if HAVE_SSE2_EXTERNAL

    if (EXTERNAL_SSE2(cpu_flags)) {

        c->put_cavs_qpel_pixels_tab[0][0] = put_cavs_qpel16_mc00_sse2;

        c->avg_cavs_qpel_pixels_tab[0][0] = avg_cavs_qpel16_mc00_sse2;

    }

#endif

}
