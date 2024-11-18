void dsputilenc_init_mmx(DSPContext* c, AVCodecContext *avctx)

{

    if (mm_flags & FF_MM_MMX) {

        const int dct_algo = avctx->dct_algo;

        if(dct_algo==FF_DCT_AUTO || dct_algo==FF_DCT_MMX){

            if(mm_flags & FF_MM_SSE2){

                c->fdct = ff_fdct_sse2;

            }else if(mm_flags & FF_MM_MMX2){

                c->fdct = ff_fdct_mmx2;

            }else{

                c->fdct = ff_fdct_mmx;

            }

        }



        c->get_pixels = get_pixels_mmx;

        c->diff_pixels = diff_pixels_mmx;

        c->pix_sum = pix_sum16_mmx;



        c->diff_bytes= diff_bytes_mmx;

        c->sum_abs_dctelem= sum_abs_dctelem_mmx;



        c->hadamard8_diff[0]= hadamard8_diff16_mmx;

        c->hadamard8_diff[1]= hadamard8_diff_mmx;



        c->pix_norm1 = pix_norm1_mmx;

        c->sse[0] = (mm_flags & FF_MM_SSE2) ? sse16_sse2 : sse16_mmx;

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





        if (mm_flags & FF_MM_MMX2) {

            c->sum_abs_dctelem= sum_abs_dctelem_mmx2;

            c->hadamard8_diff[0]= hadamard8_diff16_mmx2;

            c->hadamard8_diff[1]= hadamard8_diff_mmx2;

            c->vsad[4]= vsad_intra16_mmx2;



            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->vsad[0] = vsad16_mmx2;

            }



            c->sub_hfyu_median_prediction= sub_hfyu_median_prediction_mmx2;

        }



        if(mm_flags & FF_MM_SSE2){

            c->get_pixels = get_pixels_sse2;

            c->sum_abs_dctelem= sum_abs_dctelem_sse2;

            c->hadamard8_diff[0]= hadamard8_diff16_sse2;

            c->hadamard8_diff[1]= hadamard8_diff_sse2;

#if CONFIG_LPC

            c->lpc_compute_autocorr = ff_lpc_compute_autocorr_sse2;

#endif

        }



#if HAVE_SSSE3

        if(mm_flags & FF_MM_SSSE3){

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->try_8x8basis= try_8x8basis_ssse3;

            }

            c->add_8x8basis= add_8x8basis_ssse3;

            c->sum_abs_dctelem= sum_abs_dctelem_ssse3;

            c->hadamard8_diff[0]= hadamard8_diff16_ssse3;

            c->hadamard8_diff[1]= hadamard8_diff_ssse3;

        }

#endif



        if(mm_flags & FF_MM_3DNOW){

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->try_8x8basis= try_8x8basis_3dnow;

            }

            c->add_8x8basis= add_8x8basis_3dnow;

        }

    }



    dsputil_init_pix_mmx(c, avctx);

}
