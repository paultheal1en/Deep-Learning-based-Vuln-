void dsputil_init_mmx(DSPContext* c, AVCodecContext *avctx)

{

    mm_flags = mm_support();



    if (avctx->dsp_mask) {

        if (avctx->dsp_mask & FF_MM_FORCE)

            mm_flags |= (avctx->dsp_mask & 0xffff);

        else

            mm_flags &= ~(avctx->dsp_mask & 0xffff);

    }



#if 0

    av_log(avctx, AV_LOG_INFO, "libavcodec: CPU flags:");

    if (mm_flags & MM_MMX)

        av_log(avctx, AV_LOG_INFO, " mmx");

    if (mm_flags & MM_MMXEXT)

        av_log(avctx, AV_LOG_INFO, " mmxext");

    if (mm_flags & MM_3DNOW)

        av_log(avctx, AV_LOG_INFO, " 3dnow");

    if (mm_flags & MM_SSE)

        av_log(avctx, AV_LOG_INFO, " sse");

    if (mm_flags & MM_SSE2)

        av_log(avctx, AV_LOG_INFO, " sse2");

    av_log(avctx, AV_LOG_INFO, "\n");

#endif



    if (mm_flags & MM_MMX) {

        const int idct_algo= avctx->idct_algo;



        if(avctx->lowres==0){

            if(idct_algo==FF_IDCT_AUTO || idct_algo==FF_IDCT_SIMPLEMMX){

                c->idct_put= ff_simple_idct_put_mmx;

                c->idct_add= ff_simple_idct_add_mmx;

                c->idct    = ff_simple_idct_mmx;

                c->idct_permutation_type= FF_SIMPLE_IDCT_PERM;

#ifdef CONFIG_GPL

            }else if(idct_algo==FF_IDCT_LIBMPEG2MMX){

                if(mm_flags & MM_MMXEXT){

                    c->idct_put= ff_libmpeg2mmx2_idct_put;

                    c->idct_add= ff_libmpeg2mmx2_idct_add;

                    c->idct    = ff_mmxext_idct;

                }else{

                    c->idct_put= ff_libmpeg2mmx_idct_put;

                    c->idct_add= ff_libmpeg2mmx_idct_add;

                    c->idct    = ff_mmx_idct;

                }

                c->idct_permutation_type= FF_LIBMPEG2_IDCT_PERM;

#endif

            }else if((ENABLE_VP3_DECODER || ENABLE_VP5_DECODER || ENABLE_VP6_DECODER || ENABLE_THEORA_DECODER) &&

                     idct_algo==FF_IDCT_VP3){

                if(mm_flags & MM_SSE2){

                    c->idct_put= ff_vp3_idct_put_sse2;

                    c->idct_add= ff_vp3_idct_add_sse2;

                    c->idct    = ff_vp3_idct_sse2;

                    c->idct_permutation_type= FF_TRANSPOSE_IDCT_PERM;

                }else{

                    c->idct_put= ff_vp3_idct_put_mmx;

                    c->idct_add= ff_vp3_idct_add_mmx;

                    c->idct    = ff_vp3_idct_mmx;

                    c->idct_permutation_type= FF_PARTTRANS_IDCT_PERM;

                }

            }else if(idct_algo==FF_IDCT_CAVS){

                    c->idct_permutation_type= FF_TRANSPOSE_IDCT_PERM;

            }else if(idct_algo==FF_IDCT_XVIDMMX){

                if(mm_flags & MM_SSE2){

                    c->idct_put= ff_idct_xvid_sse2_put;

                    c->idct_add= ff_idct_xvid_sse2_add;

                    c->idct    = ff_idct_xvid_sse2;

                    c->idct_permutation_type= FF_SSE2_IDCT_PERM;

                }else if(mm_flags & MM_MMXEXT){

                    c->idct_put= ff_idct_xvid_mmx2_put;

                    c->idct_add= ff_idct_xvid_mmx2_add;

                    c->idct    = ff_idct_xvid_mmx2;

                }else{

                    c->idct_put= ff_idct_xvid_mmx_put;

                    c->idct_add= ff_idct_xvid_mmx_add;

                    c->idct    = ff_idct_xvid_mmx;

                }

            }

        }



        c->put_pixels_clamped = put_pixels_clamped_mmx;

        c->put_signed_pixels_clamped = put_signed_pixels_clamped_mmx;

        c->add_pixels_clamped = add_pixels_clamped_mmx;

        c->clear_blocks = clear_blocks_mmx;



#define SET_HPEL_FUNCS(PFX, IDX, SIZE, CPU) \

        c->PFX ## _pixels_tab[IDX][0] = PFX ## _pixels ## SIZE ## _ ## CPU; \

        c->PFX ## _pixels_tab[IDX][1] = PFX ## _pixels ## SIZE ## _x2_ ## CPU; \

        c->PFX ## _pixels_tab[IDX][2] = PFX ## _pixels ## SIZE ## _y2_ ## CPU; \

        c->PFX ## _pixels_tab[IDX][3] = PFX ## _pixels ## SIZE ## _xy2_ ## CPU



        SET_HPEL_FUNCS(put, 0, 16, mmx);

        SET_HPEL_FUNCS(put_no_rnd, 0, 16, mmx);

        SET_HPEL_FUNCS(avg, 0, 16, mmx);

        SET_HPEL_FUNCS(avg_no_rnd, 0, 16, mmx);

        SET_HPEL_FUNCS(put, 1, 8, mmx);

        SET_HPEL_FUNCS(put_no_rnd, 1, 8, mmx);

        SET_HPEL_FUNCS(avg, 1, 8, mmx);

        SET_HPEL_FUNCS(avg_no_rnd, 1, 8, mmx);



        c->gmc= gmc_mmx;



        c->add_bytes= add_bytes_mmx;

        c->add_bytes_l2= add_bytes_l2_mmx;



        c->draw_edges = draw_edges_mmx;



        if (ENABLE_ANY_H263) {

            c->h263_v_loop_filter= h263_v_loop_filter_mmx;

            c->h263_h_loop_filter= h263_h_loop_filter_mmx;

        }

        if ((ENABLE_VP3_DECODER || ENABLE_THEORA_DECODER) &&

            !(avctx->flags & CODEC_FLAG_BITEXACT)) {

            c->vp3_v_loop_filter= ff_vp3_v_loop_filter_mmx;

            c->vp3_h_loop_filter= ff_vp3_h_loop_filter_mmx;

        }

        c->put_h264_chroma_pixels_tab[0]= put_h264_chroma_mc8_mmx_rnd;

        c->put_h264_chroma_pixels_tab[1]= put_h264_chroma_mc4_mmx;

        c->put_no_rnd_h264_chroma_pixels_tab[0]= put_h264_chroma_mc8_mmx_nornd;



        c->h264_idct_dc_add=

        c->h264_idct_add= ff_h264_idct_add_mmx;

        c->h264_idct8_dc_add=

        c->h264_idct8_add= ff_h264_idct8_add_mmx;

        if (mm_flags & MM_SSE2)

            c->h264_idct8_add= ff_h264_idct8_add_sse2;



        if (mm_flags & MM_MMXEXT) {

            c->prefetch = prefetch_mmx2;



            c->put_pixels_tab[0][1] = put_pixels16_x2_mmx2;

            c->put_pixels_tab[0][2] = put_pixels16_y2_mmx2;



            c->avg_pixels_tab[0][0] = avg_pixels16_mmx2;

            c->avg_pixels_tab[0][1] = avg_pixels16_x2_mmx2;

            c->avg_pixels_tab[0][2] = avg_pixels16_y2_mmx2;



            c->put_pixels_tab[1][1] = put_pixels8_x2_mmx2;

            c->put_pixels_tab[1][2] = put_pixels8_y2_mmx2;



            c->avg_pixels_tab[1][0] = avg_pixels8_mmx2;

            c->avg_pixels_tab[1][1] = avg_pixels8_x2_mmx2;

            c->avg_pixels_tab[1][2] = avg_pixels8_y2_mmx2;



            c->h264_idct_dc_add= ff_h264_idct_dc_add_mmx2;

            c->h264_idct8_dc_add= ff_h264_idct8_dc_add_mmx2;



            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->put_no_rnd_pixels_tab[0][1] = put_no_rnd_pixels16_x2_mmx2;

                c->put_no_rnd_pixels_tab[0][2] = put_no_rnd_pixels16_y2_mmx2;

                c->put_no_rnd_pixels_tab[1][1] = put_no_rnd_pixels8_x2_mmx2;

                c->put_no_rnd_pixels_tab[1][2] = put_no_rnd_pixels8_y2_mmx2;

                c->avg_pixels_tab[0][3] = avg_pixels16_xy2_mmx2;

                c->avg_pixels_tab[1][3] = avg_pixels8_xy2_mmx2;

            }



#define SET_QPEL_FUNCS(PFX, IDX, SIZE, CPU) \

            c->PFX ## _pixels_tab[IDX][ 0] = PFX ## SIZE ## _mc00_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 1] = PFX ## SIZE ## _mc10_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 2] = PFX ## SIZE ## _mc20_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 3] = PFX ## SIZE ## _mc30_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 4] = PFX ## SIZE ## _mc01_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 5] = PFX ## SIZE ## _mc11_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 6] = PFX ## SIZE ## _mc21_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 7] = PFX ## SIZE ## _mc31_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 8] = PFX ## SIZE ## _mc02_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][ 9] = PFX ## SIZE ## _mc12_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][10] = PFX ## SIZE ## _mc22_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][11] = PFX ## SIZE ## _mc32_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][12] = PFX ## SIZE ## _mc03_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][13] = PFX ## SIZE ## _mc13_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][14] = PFX ## SIZE ## _mc23_ ## CPU; \

            c->PFX ## _pixels_tab[IDX][15] = PFX ## SIZE ## _mc33_ ## CPU



            SET_QPEL_FUNCS(put_qpel, 0, 16, mmx2);

            SET_QPEL_FUNCS(put_qpel, 1, 8, mmx2);

            SET_QPEL_FUNCS(put_no_rnd_qpel, 0, 16, mmx2);

            SET_QPEL_FUNCS(put_no_rnd_qpel, 1, 8, mmx2);

            SET_QPEL_FUNCS(avg_qpel, 0, 16, mmx2);

            SET_QPEL_FUNCS(avg_qpel, 1, 8, mmx2);



            SET_QPEL_FUNCS(put_h264_qpel, 0, 16, mmx2);

            SET_QPEL_FUNCS(put_h264_qpel, 1, 8, mmx2);

            SET_QPEL_FUNCS(put_h264_qpel, 2, 4, mmx2);

            SET_QPEL_FUNCS(avg_h264_qpel, 0, 16, mmx2);

            SET_QPEL_FUNCS(avg_h264_qpel, 1, 8, mmx2);

            SET_QPEL_FUNCS(avg_h264_qpel, 2, 4, mmx2);



            SET_QPEL_FUNCS(put_2tap_qpel, 0, 16, mmx2);

            SET_QPEL_FUNCS(put_2tap_qpel, 1, 8, mmx2);

            SET_QPEL_FUNCS(avg_2tap_qpel, 0, 16, mmx2);

            SET_QPEL_FUNCS(avg_2tap_qpel, 1, 8, mmx2);



            c->avg_h264_chroma_pixels_tab[0]= avg_h264_chroma_mc8_mmx2_rnd;

            c->avg_h264_chroma_pixels_tab[1]= avg_h264_chroma_mc4_mmx2;

            c->avg_h264_chroma_pixels_tab[2]= avg_h264_chroma_mc2_mmx2;

            c->put_h264_chroma_pixels_tab[2]= put_h264_chroma_mc2_mmx2;

            c->h264_v_loop_filter_luma= h264_v_loop_filter_luma_mmx2;

            c->h264_h_loop_filter_luma= h264_h_loop_filter_luma_mmx2;

            c->h264_v_loop_filter_chroma= h264_v_loop_filter_chroma_mmx2;

            c->h264_h_loop_filter_chroma= h264_h_loop_filter_chroma_mmx2;

            c->h264_v_loop_filter_chroma_intra= h264_v_loop_filter_chroma_intra_mmx2;

            c->h264_h_loop_filter_chroma_intra= h264_h_loop_filter_chroma_intra_mmx2;

            c->h264_loop_filter_strength= h264_loop_filter_strength_mmx2;



            c->weight_h264_pixels_tab[0]= ff_h264_weight_16x16_mmx2;

            c->weight_h264_pixels_tab[1]= ff_h264_weight_16x8_mmx2;

            c->weight_h264_pixels_tab[2]= ff_h264_weight_8x16_mmx2;

            c->weight_h264_pixels_tab[3]= ff_h264_weight_8x8_mmx2;

            c->weight_h264_pixels_tab[4]= ff_h264_weight_8x4_mmx2;

            c->weight_h264_pixels_tab[5]= ff_h264_weight_4x8_mmx2;

            c->weight_h264_pixels_tab[6]= ff_h264_weight_4x4_mmx2;

            c->weight_h264_pixels_tab[7]= ff_h264_weight_4x2_mmx2;



            c->biweight_h264_pixels_tab[0]= ff_h264_biweight_16x16_mmx2;

            c->biweight_h264_pixels_tab[1]= ff_h264_biweight_16x8_mmx2;

            c->biweight_h264_pixels_tab[2]= ff_h264_biweight_8x16_mmx2;

            c->biweight_h264_pixels_tab[3]= ff_h264_biweight_8x8_mmx2;

            c->biweight_h264_pixels_tab[4]= ff_h264_biweight_8x4_mmx2;

            c->biweight_h264_pixels_tab[5]= ff_h264_biweight_4x8_mmx2;

            c->biweight_h264_pixels_tab[6]= ff_h264_biweight_4x4_mmx2;

            c->biweight_h264_pixels_tab[7]= ff_h264_biweight_4x2_mmx2;



            if (ENABLE_CAVS_DECODER)

                ff_cavsdsp_init_mmx2(c, avctx);



            if (ENABLE_VC1_DECODER || ENABLE_WMV3_DECODER)

                ff_vc1dsp_init_mmx(c, avctx);



            c->add_png_paeth_prediction= add_png_paeth_prediction_mmx2;

        } else if (mm_flags & MM_3DNOW) {

            c->prefetch = prefetch_3dnow;



            c->put_pixels_tab[0][1] = put_pixels16_x2_3dnow;

            c->put_pixels_tab[0][2] = put_pixels16_y2_3dnow;



            c->avg_pixels_tab[0][0] = avg_pixels16_3dnow;

            c->avg_pixels_tab[0][1] = avg_pixels16_x2_3dnow;

            c->avg_pixels_tab[0][2] = avg_pixels16_y2_3dnow;



            c->put_pixels_tab[1][1] = put_pixels8_x2_3dnow;

            c->put_pixels_tab[1][2] = put_pixels8_y2_3dnow;



            c->avg_pixels_tab[1][0] = avg_pixels8_3dnow;

            c->avg_pixels_tab[1][1] = avg_pixels8_x2_3dnow;

            c->avg_pixels_tab[1][2] = avg_pixels8_y2_3dnow;



            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->put_no_rnd_pixels_tab[0][1] = put_no_rnd_pixels16_x2_3dnow;

                c->put_no_rnd_pixels_tab[0][2] = put_no_rnd_pixels16_y2_3dnow;

                c->put_no_rnd_pixels_tab[1][1] = put_no_rnd_pixels8_x2_3dnow;

                c->put_no_rnd_pixels_tab[1][2] = put_no_rnd_pixels8_y2_3dnow;

                c->avg_pixels_tab[0][3] = avg_pixels16_xy2_3dnow;

                c->avg_pixels_tab[1][3] = avg_pixels8_xy2_3dnow;

            }



            SET_QPEL_FUNCS(put_qpel, 0, 16, 3dnow);

            SET_QPEL_FUNCS(put_qpel, 1, 8, 3dnow);

            SET_QPEL_FUNCS(put_no_rnd_qpel, 0, 16, 3dnow);

            SET_QPEL_FUNCS(put_no_rnd_qpel, 1, 8, 3dnow);

            SET_QPEL_FUNCS(avg_qpel, 0, 16, 3dnow);

            SET_QPEL_FUNCS(avg_qpel, 1, 8, 3dnow);



            SET_QPEL_FUNCS(put_h264_qpel, 0, 16, 3dnow);

            SET_QPEL_FUNCS(put_h264_qpel, 1, 8, 3dnow);

            SET_QPEL_FUNCS(put_h264_qpel, 2, 4, 3dnow);

            SET_QPEL_FUNCS(avg_h264_qpel, 0, 16, 3dnow);

            SET_QPEL_FUNCS(avg_h264_qpel, 1, 8, 3dnow);

            SET_QPEL_FUNCS(avg_h264_qpel, 2, 4, 3dnow);



            SET_QPEL_FUNCS(put_2tap_qpel, 0, 16, 3dnow);

            SET_QPEL_FUNCS(put_2tap_qpel, 1, 8, 3dnow);

            SET_QPEL_FUNCS(avg_2tap_qpel, 0, 16, 3dnow);

            SET_QPEL_FUNCS(avg_2tap_qpel, 1, 8, 3dnow);



            c->avg_h264_chroma_pixels_tab[0]= avg_h264_chroma_mc8_3dnow_rnd;

            c->avg_h264_chroma_pixels_tab[1]= avg_h264_chroma_mc4_3dnow;



            if (ENABLE_CAVS_DECODER)

                ff_cavsdsp_init_3dnow(c, avctx);

        }





#define H264_QPEL_FUNCS(x, y, CPU)\

            c->put_h264_qpel_pixels_tab[0][x+y*4] = put_h264_qpel16_mc##x##y##_##CPU;\

            c->put_h264_qpel_pixels_tab[1][x+y*4] = put_h264_qpel8_mc##x##y##_##CPU;\

            c->avg_h264_qpel_pixels_tab[0][x+y*4] = avg_h264_qpel16_mc##x##y##_##CPU;\

            c->avg_h264_qpel_pixels_tab[1][x+y*4] = avg_h264_qpel8_mc##x##y##_##CPU;

        if((mm_flags & MM_SSE2) && !(mm_flags & MM_3DNOW)){

            // these functions are slower than mmx on AMD, but faster on Intel

/* FIXME works in most codecs, but crashes svq1 due to unaligned chroma

            c->put_pixels_tab[0][0] = put_pixels16_sse2;

            c->avg_pixels_tab[0][0] = avg_pixels16_sse2;

*/

            H264_QPEL_FUNCS(0, 0, sse2);

        }

        if(mm_flags & MM_SSE2){

            H264_QPEL_FUNCS(0, 1, sse2);

            H264_QPEL_FUNCS(0, 2, sse2);

            H264_QPEL_FUNCS(0, 3, sse2);

            H264_QPEL_FUNCS(1, 1, sse2);

            H264_QPEL_FUNCS(1, 2, sse2);

            H264_QPEL_FUNCS(1, 3, sse2);

            H264_QPEL_FUNCS(2, 1, sse2);

            H264_QPEL_FUNCS(2, 2, sse2);

            H264_QPEL_FUNCS(2, 3, sse2);

            H264_QPEL_FUNCS(3, 1, sse2);

            H264_QPEL_FUNCS(3, 2, sse2);

            H264_QPEL_FUNCS(3, 3, sse2);

        }

#ifdef HAVE_SSSE3

        if(mm_flags & MM_SSSE3){

            H264_QPEL_FUNCS(1, 0, ssse3);

            H264_QPEL_FUNCS(1, 1, ssse3);

            H264_QPEL_FUNCS(1, 2, ssse3);

            H264_QPEL_FUNCS(1, 3, ssse3);

            H264_QPEL_FUNCS(2, 0, ssse3);

            H264_QPEL_FUNCS(2, 1, ssse3);

            H264_QPEL_FUNCS(2, 2, ssse3);

            H264_QPEL_FUNCS(2, 3, ssse3);

            H264_QPEL_FUNCS(3, 0, ssse3);

            H264_QPEL_FUNCS(3, 1, ssse3);

            H264_QPEL_FUNCS(3, 2, ssse3);

            H264_QPEL_FUNCS(3, 3, ssse3);

            c->put_no_rnd_h264_chroma_pixels_tab[0]= put_h264_chroma_mc8_ssse3_nornd;

            c->put_h264_chroma_pixels_tab[0]= put_h264_chroma_mc8_ssse3_rnd;

            c->avg_h264_chroma_pixels_tab[0]= avg_h264_chroma_mc8_ssse3_rnd;

            c->put_h264_chroma_pixels_tab[1]= put_h264_chroma_mc4_ssse3;

            c->avg_h264_chroma_pixels_tab[1]= avg_h264_chroma_mc4_ssse3;

            c->add_png_paeth_prediction= add_png_paeth_prediction_ssse3;

        }

#endif



#ifdef CONFIG_SNOW_DECODER

        if(mm_flags & MM_SSE2 & 0){

            c->horizontal_compose97i = ff_snow_horizontal_compose97i_sse2;

#ifdef HAVE_7REGS

            c->vertical_compose97i = ff_snow_vertical_compose97i_sse2;

#endif

            c->inner_add_yblock = ff_snow_inner_add_yblock_sse2;

        }

        else{

            if(mm_flags & MM_MMXEXT){

            c->horizontal_compose97i = ff_snow_horizontal_compose97i_mmx;

#ifdef HAVE_7REGS

            c->vertical_compose97i = ff_snow_vertical_compose97i_mmx;

#endif

            }

            c->inner_add_yblock = ff_snow_inner_add_yblock_mmx;

        }

#endif



        if(mm_flags & MM_3DNOW){

            c->vorbis_inverse_coupling = vorbis_inverse_coupling_3dnow;

            c->vector_fmul = vector_fmul_3dnow;

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->float_to_int16 = float_to_int16_3dnow;

                c->float_to_int16_interleave = float_to_int16_interleave_3dnow;

            }

        }

        if(mm_flags & MM_3DNOWEXT){

            c->vector_fmul_reverse = vector_fmul_reverse_3dnow2;

            c->vector_fmul_window = vector_fmul_window_3dnow2;

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){

                c->float_to_int16_interleave = float_to_int16_interleave_3dn2;

            }

        }

        if(mm_flags & MM_SSE){

            c->vorbis_inverse_coupling = vorbis_inverse_coupling_sse;

            c->ac3_downmix = ac3_downmix_sse;

            c->vector_fmul = vector_fmul_sse;

            c->vector_fmul_reverse = vector_fmul_reverse_sse;

            c->vector_fmul_add_add = vector_fmul_add_add_sse;

            c->vector_fmul_window = vector_fmul_window_sse;

            c->int32_to_float_fmul_scalar = int32_to_float_fmul_scalar_sse;

            c->float_to_int16 = float_to_int16_sse;

            c->float_to_int16_interleave = float_to_int16_interleave_sse;

        }

        if(mm_flags & MM_3DNOW)

            c->vector_fmul_add_add = vector_fmul_add_add_3dnow; // faster than sse

        if(mm_flags & MM_SSE2){

            c->int32_to_float_fmul_scalar = int32_to_float_fmul_scalar_sse2;

            c->float_to_int16 = float_to_int16_sse2;

            c->float_to_int16_interleave = float_to_int16_interleave_sse2;

            c->add_int16 = add_int16_sse2;

            c->sub_int16 = sub_int16_sse2;

            c->scalarproduct_int16 = scalarproduct_int16_sse2;

        }

    }



    if (ENABLE_ENCODERS)

        dsputilenc_init_mmx(c, avctx);



#if 0

    // for speed testing

    get_pixels = just_return;

    put_pixels_clamped = just_return;

    add_pixels_clamped = just_return;



    pix_abs16x16 = just_return;

    pix_abs16x16_x2 = just_return;

    pix_abs16x16_y2 = just_return;

    pix_abs16x16_xy2 = just_return;



    put_pixels_tab[0] = just_return;

    put_pixels_tab[1] = just_return;

    put_pixels_tab[2] = just_return;

    put_pixels_tab[3] = just_return;



    put_no_rnd_pixels_tab[0] = just_return;

    put_no_rnd_pixels_tab[1] = just_return;

    put_no_rnd_pixels_tab[2] = just_return;

    put_no_rnd_pixels_tab[3] = just_return;



    avg_pixels_tab[0] = just_return;

    avg_pixels_tab[1] = just_return;

    avg_pixels_tab[2] = just_return;

    avg_pixels_tab[3] = just_return;



    avg_no_rnd_pixels_tab[0] = just_return;

    avg_no_rnd_pixels_tab[1] = just_return;

    avg_no_rnd_pixels_tab[2] = just_return;

    avg_no_rnd_pixels_tab[3] = just_return;



    //av_fdct = just_return;

    //ff_idct = just_return;

#endif

}
