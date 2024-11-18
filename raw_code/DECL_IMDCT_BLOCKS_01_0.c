DECL_IMDCT_BLOCKS(sse,sse)

DECL_IMDCT_BLOCKS(sse2,sse)

DECL_IMDCT_BLOCKS(sse3,sse)

DECL_IMDCT_BLOCKS(ssse3,sse)

DECL_IMDCT_BLOCKS(avx,avx)

#endif /* HAVE_YASM */



void ff_mpadsp_init_mmx(MPADSPContext *s)

{

    int mm_flags = av_get_cpu_flags();



    int i, j;

    for (j = 0; j < 4; j++) {

        for (i = 0; i < 40; i ++) {

            mdct_win_sse[0][j][4*i    ] = ff_mdct_win_float[j    ][i];

            mdct_win_sse[0][j][4*i + 1] = ff_mdct_win_float[j + 4][i];

            mdct_win_sse[0][j][4*i + 2] = ff_mdct_win_float[j    ][i];

            mdct_win_sse[0][j][4*i + 3] = ff_mdct_win_float[j + 4][i];

            mdct_win_sse[1][j][4*i    ] = ff_mdct_win_float[0    ][i];

            mdct_win_sse[1][j][4*i + 1] = ff_mdct_win_float[4    ][i];

            mdct_win_sse[1][j][4*i + 2] = ff_mdct_win_float[j    ][i];

            mdct_win_sse[1][j][4*i + 3] = ff_mdct_win_float[j + 4][i];

        }

    }



#if HAVE_SSE2_INLINE

    if (mm_flags & AV_CPU_FLAG_SSE2) {

        s->apply_window_float = apply_window_mp3;

    }

#endif /* HAVE_SSE2_INLINE */



#if HAVE_YASM

    if (mm_flags & AV_CPU_FLAG_AVX && HAVE_AVX) {

        s->imdct36_blocks_float = imdct36_blocks_avx;

#if HAVE_SSE

    } else if (mm_flags & AV_CPU_FLAG_SSSE3) {

        s->imdct36_blocks_float = imdct36_blocks_ssse3;

    } else if (mm_flags & AV_CPU_FLAG_SSE3) {

        s->imdct36_blocks_float = imdct36_blocks_sse3;

    } else if (mm_flags & AV_CPU_FLAG_SSE2) {

        s->imdct36_blocks_float = imdct36_blocks_sse2;

    } else if (mm_flags & AV_CPU_FLAG_SSE) {

        s->imdct36_blocks_float = imdct36_blocks_sse;

#endif /* HAVE_SSE */

    }

#endif /* HAVE_YASM */

}
