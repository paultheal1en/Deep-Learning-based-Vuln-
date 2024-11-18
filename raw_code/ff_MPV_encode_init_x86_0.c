void ff_MPV_encode_init_x86(MpegEncContext *s)

{

    int mm_flags = av_get_cpu_flags();

    const int dct_algo = s->avctx->dct_algo;



    if (dct_algo == FF_DCT_AUTO || dct_algo == FF_DCT_MMX) {

#if HAVE_MMX_INLINE

        if (mm_flags & AV_CPU_FLAG_MMX && HAVE_MMX)

            s->dct_quantize = dct_quantize_MMX;

#endif

#if HAVE_MMXEXT_INLINE

        if (mm_flags & AV_CPU_FLAG_MMXEXT && HAVE_MMXEXT)

            s->dct_quantize = dct_quantize_MMX2;

#endif

#if HAVE_SSE2_INLINE

        if (mm_flags & AV_CPU_FLAG_SSE2 && HAVE_SSE2)

            s->dct_quantize = dct_quantize_SSE2;

#endif

#if HAVE_SSSE3_INLINE

        if (mm_flags & AV_CPU_FLAG_SSSE3)

            s->dct_quantize = dct_quantize_SSSE3;

#endif

    }

}
