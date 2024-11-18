hadamard_func(mmxext)

hadamard_func(sse2)

hadamard_func(ssse3)



av_cold void ff_dsputilenc_init_mmx(DSPContext *c, AVCodecContext *avctx)

{

    int cpu_flags = av_get_cpu_flags();



#if HAVE_YASM

    int bit_depth = avctx->bits_per_raw_sample;



    if (EXTERNAL_MMX(cpu_flags)) {

        if (bit_depth <= 8)

            c->get_pixels = ff_get_pixels_mmx;

        c->diff_pixels = ff_diff_pixels_mmx;

        c->pix_sum = ff_pix_sum16_mmx;



        c->pix_norm1 = ff_pix_norm1_mmx;

    }

    if (EXTERNAL_SSE2(cpu_flags))

        if (bit_depth <= 8)

            c->get_pixels = ff_get_pixels_sse2;

#endif /* HAVE_YASM */



#if HAVE_INLINE_ASM

    if (cpu_flags & AV_CPU_FLAG_MMX) {

        const int dct_algo = avctx->dct_algo;

        if (avctx->bits_per_raw_sample <= 8 &&

            (dct_algo==FF_DCT_AUTO || dct_algo==FF_DCT_MMX)) {

            if (cpu_flags & AV_CPU_FLAG_SSE2) {

                c->fdct = ff_fdct_sse2;

            } else if (cpu_flags & AV_CPU_FLAG_MMXEXT) {

                c->fdct = ff_fdct_mmxext;

            }else{

                c->fdct = ff_fdct_mmx;

            }

        }





        c->diff_bytes= diff_bytes_mmx;

        c->sum_abs_dctelem= sum_abs_dctelem_mmx;



        c->sse[0] = sse16_mmx;

        c->sse[1] = sse8_mmx;

        c->vsad[4]= vsad_intra16_mmx;



        c->nsse[0] = nsse16_mmx;

        c->nsse[1] = nsse8_mmx;

        if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

            c->vsad[0] = vsad16_mmx;

        }



        if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

            c->try_8x8basis= try_8x8basis_mmx;

        }

        c->add_8x8basis= add_8x8basis_mmx;



        c->ssd_int8_vs_int16 = ssd_int8_vs_int16_mmx;



        if (cpu_flags & AV_CPU_FLAG_MMXEXT) {

            c->sum_abs_dctelem = sum_abs_dctelem_mmxext;

            c->vsad[4]         = vsad_intra16_mmxext;



            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->vsad[0] = vsad16_mmxext;

            }



            c->sub_hfyu_median_prediction = sub_hfyu_median_prediction_mmxext;

        }



        if (cpu_flags & AV_CPU_FLAG_SSE2) {

            c->sum_abs_dctelem= sum_abs_dctelem_sse2;

        }



#if HAVE_SSSE3_INLINE

        if (cpu_flags & AV_CPU_FLAG_SSSE3) {

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->try_8x8basis= try_8x8basis_ssse3;

            }

            c->add_8x8basis= add_8x8basis_ssse3;

            c->sum_abs_dctelem= sum_abs_dctelem_ssse3;

        }

#endif



        if (cpu_flags & AV_CPU_FLAG_3DNOW) {

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->try_8x8basis= try_8x8basis_3dnow;

            }

            c->add_8x8basis= add_8x8basis_3dnow;

        }

    }

#endif /* HAVE_INLINE_ASM */



    if (EXTERNAL_MMX(cpu_flags)) {

        c->hadamard8_diff[0] = ff_hadamard8_diff16_mmx;

        c->hadamard8_diff[1] = ff_hadamard8_diff_mmx;



        if (EXTERNAL_MMXEXT(cpu_flags)) {

            c->hadamard8_diff[0] = ff_hadamard8_diff16_mmxext;

            c->hadamard8_diff[1] = ff_hadamard8_diff_mmxext;

        }



        if (EXTERNAL_SSE2(cpu_flags)) {

            c->sse[0] = ff_sse16_sse2;



#if HAVE_ALIGNED_STACK

            c->hadamard8_diff[0] = ff_hadamard8_diff16_sse2;

            c->hadamard8_diff[1] = ff_hadamard8_diff_sse2;

#endif

        }



        if (EXTERNAL_SSSE3(cpu_flags) && HAVE_ALIGNED_STACK) {

            c->hadamard8_diff[0] = ff_hadamard8_diff16_ssse3;

            c->hadamard8_diff[1] = ff_hadamard8_diff_ssse3;

        }

    }



    ff_dsputil_init_pix_mmx(c, avctx);

}
