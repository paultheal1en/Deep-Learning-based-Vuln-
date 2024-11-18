void ff_h264_pred_init_x86(H264PredContext *h, int codec_id)

{

    mm_flags = mm_support();



#if HAVE_YASM

    if (mm_flags & FF_MM_MMX) {

        h->pred16x16[VERT_PRED8x8] = ff_pred16x16_vertical_mmx;

        h->pred16x16[HOR_PRED8x8 ] = ff_pred16x16_horizontal_mmx;

        h->pred8x8  [VERT_PRED8x8] = ff_pred8x8_vertical_mmx;

        h->pred8x8  [HOR_PRED8x8 ] = ff_pred8x8_horizontal_mmx;

        if (codec_id == CODEC_ID_VP8) {

            h->pred16x16[PLANE_PRED8x8] = ff_pred16x16_tm_vp8_mmx;

            h->pred8x8  [PLANE_PRED8x8] = ff_pred8x8_tm_vp8_mmx;

            h->pred4x4  [TM_VP8_PRED  ] = ff_pred4x4_tm_vp8_mmx;

        }

    }



    if (mm_flags & FF_MM_MMX2) {

        h->pred16x16[HOR_PRED8x8 ] = ff_pred16x16_horizontal_mmxext;

        h->pred16x16[DC_PRED8x8  ] = ff_pred16x16_dc_mmxext;

        h->pred8x8  [HOR_PRED8x8 ] = ff_pred8x8_horizontal_mmxext;

        h->pred4x4  [DC_PRED     ] = ff_pred4x4_dc_mmxext;

        if (codec_id == CODEC_ID_VP8) {

            h->pred16x16[PLANE_PRED8x8] = ff_pred16x16_tm_vp8_mmxext;

            h->pred8x8  [DC_PRED8x8   ] = ff_pred8x8_dc_rv40_mmxext;

            h->pred8x8  [PLANE_PRED8x8] = ff_pred8x8_tm_vp8_mmxext;

            h->pred4x4  [TM_VP8_PRED  ] = ff_pred4x4_tm_vp8_mmxext;

            h->pred4x4  [VERT_PRED    ] = ff_pred4x4_vertical_vp8_mmxext;

        }

    }



    if (mm_flags & FF_MM_SSE) {

        h->pred16x16[VERT_PRED8x8] = ff_pred16x16_vertical_sse;

        h->pred16x16[DC_PRED8x8  ] = ff_pred16x16_dc_sse;

    }



    if (mm_flags & FF_MM_SSE2) {

        h->pred16x16[DC_PRED8x8  ] = ff_pred16x16_dc_sse2;

        if (codec_id == CODEC_ID_VP8) {

            h->pred16x16[PLANE_PRED8x8] = ff_pred16x16_tm_vp8_sse2;

            h->pred8x8  [PLANE_PRED8x8] = ff_pred8x8_tm_vp8_sse2;

        }

    }



    if (mm_flags & FF_MM_SSSE3) {

        h->pred16x16[HOR_PRED8x8 ] = ff_pred16x16_horizontal_ssse3;

        h->pred16x16[DC_PRED8x8  ] = ff_pred16x16_dc_ssse3;

        h->pred8x8  [HOR_PRED8x8 ] = ff_pred8x8_horizontal_ssse3;

        if (codec_id == CODEC_ID_VP8) {

            h->pred8x8  [PLANE_PRED8x8] = ff_pred8x8_tm_vp8_ssse3;

            h->pred4x4  [TM_VP8_PRED  ] = ff_pred4x4_tm_vp8_ssse3;

        }

    }

#endif

}
