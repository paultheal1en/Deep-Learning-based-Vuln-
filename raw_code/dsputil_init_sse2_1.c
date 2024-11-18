static av_cold void dsputil_init_sse2(DSPContext *c, AVCodecContext *avctx,

                                      int mm_flags)

{

#if HAVE_SSE2_INLINE

    const int high_bit_depth = avctx->bits_per_raw_sample > 8;



    if (!high_bit_depth && avctx->idct_algo == FF_IDCT_XVIDMMX) {

        c->idct_put              = ff_idct_xvid_sse2_put;

        c->idct_add              = ff_idct_xvid_sse2_add;

        c->idct                  = ff_idct_xvid_sse2;

        c->idct_permutation_type = FF_SSE2_IDCT_PERM;

    }

#endif /* HAVE_SSE2_INLINE */



#if HAVE_SSE2_EXTERNAL

    c->scalarproduct_int16          = ff_scalarproduct_int16_sse2;

    c->scalarproduct_and_madd_int16 = ff_scalarproduct_and_madd_int16_sse2;

    if (mm_flags & AV_CPU_FLAG_ATOM) {

        c->vector_clip_int32 = ff_vector_clip_int32_int_sse2;

    } else {

        c->vector_clip_int32 = ff_vector_clip_int32_sse2;

    }

    if (avctx->flags & CODEC_FLAG_BITEXACT) {

        c->apply_window_int16 = ff_apply_window_int16_sse2;

    } else if (!(mm_flags & AV_CPU_FLAG_SSE2SLOW)) {

        c->apply_window_int16 = ff_apply_window_int16_round_sse2;

    }

    c->bswap_buf = ff_bswap32_buf_sse2;

#endif /* HAVE_SSE2_EXTERNAL */

}
