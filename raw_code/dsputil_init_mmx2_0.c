static void dsputil_init_mmx2(DSPContext *c, AVCodecContext *avctx,

                              int mm_flags)

{

    const int bit_depth      = avctx->bits_per_raw_sample;

    const int high_bit_depth = bit_depth > 8;



    c->prefetch = prefetch_mmx2;



    if (!high_bit_depth) {

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

    }



    if (!(avctx->flags & CODEC_FLAG_BITEXACT)) {

        if (!high_bit_depth) {

            c->put_no_rnd_pixels_tab[0][1] = put_no_rnd_pixels16_x2_mmx2;

            c->put_no_rnd_pixels_tab[0][2] = put_no_rnd_pixels16_y2_mmx2;

            c->put_no_rnd_pixels_tab[1][1] = put_no_rnd_pixels8_x2_mmx2;

            c->put_no_rnd_pixels_tab[1][2] = put_no_rnd_pixels8_y2_mmx2;



            c->avg_pixels_tab[0][3] = avg_pixels16_xy2_mmx2;

            c->avg_pixels_tab[1][3] = avg_pixels8_xy2_mmx2;

        }



        if (CONFIG_VP3_DECODER && HAVE_YASM) {

            c->vp3_v_loop_filter = ff_vp3_v_loop_filter_mmx2;

            c->vp3_h_loop_filter = ff_vp3_h_loop_filter_mmx2;

        }

    }

    if (CONFIG_VP3_DECODER && HAVE_YASM)

        c->vp3_idct_dc_add = ff_vp3_idct_dc_add_mmx2;



    if (CONFIG_VP3_DECODER && (avctx->codec_id == CODEC_ID_VP3 ||

                               avctx->codec_id == CODEC_ID_THEORA)) {

        c->put_no_rnd_pixels_tab[1][1] = put_no_rnd_pixels8_x2_exact_mmx2;

        c->put_no_rnd_pixels_tab[1][2] = put_no_rnd_pixels8_y2_exact_mmx2;

    }



    if (CONFIG_H264QPEL) {

        SET_QPEL_FUNCS(put_qpel,        0, 16, mmx2, );

        SET_QPEL_FUNCS(put_qpel,        1,  8, mmx2, );

        SET_QPEL_FUNCS(put_no_rnd_qpel, 0, 16, mmx2, );

        SET_QPEL_FUNCS(put_no_rnd_qpel, 1,  8, mmx2, );

        SET_QPEL_FUNCS(avg_qpel,        0, 16, mmx2, );

        SET_QPEL_FUNCS(avg_qpel,        1,  8, mmx2, );



        if (!high_bit_depth) {

            SET_QPEL_FUNCS(put_h264_qpel, 0, 16, mmx2, );

            SET_QPEL_FUNCS(put_h264_qpel, 1,  8, mmx2, );

            SET_QPEL_FUNCS(put_h264_qpel, 2,  4, mmx2, );

            SET_QPEL_FUNCS(avg_h264_qpel, 0, 16, mmx2, );

            SET_QPEL_FUNCS(avg_h264_qpel, 1,  8, mmx2, );

            SET_QPEL_FUNCS(avg_h264_qpel, 2,  4, mmx2, );

        } else if (bit_depth == 10) {

#if HAVE_YASM

#if !ARCH_X86_64

            SET_QPEL_FUNCS(avg_h264_qpel, 0, 16, 10_mmxext, ff_);

            SET_QPEL_FUNCS(put_h264_qpel, 0, 16, 10_mmxext, ff_);

            SET_QPEL_FUNCS(put_h264_qpel, 1,  8, 10_mmxext, ff_);

            SET_QPEL_FUNCS(avg_h264_qpel, 1,  8, 10_mmxext, ff_);

#endif

            SET_QPEL_FUNCS(put_h264_qpel, 2, 4,  10_mmxext, ff_);

            SET_QPEL_FUNCS(avg_h264_qpel, 2, 4,  10_mmxext, ff_);

#endif

        }



        SET_QPEL_FUNCS(put_2tap_qpel, 0, 16, mmx2, );

        SET_QPEL_FUNCS(put_2tap_qpel, 1,  8, mmx2, );

        SET_QPEL_FUNCS(avg_2tap_qpel, 0, 16, mmx2, );

        SET_QPEL_FUNCS(avg_2tap_qpel, 1,  8, mmx2, );

    }



#if HAVE_YASM

    if (!high_bit_depth && CONFIG_H264CHROMA) {

        c->avg_h264_chroma_pixels_tab[0] = ff_avg_h264_chroma_mc8_mmx2_rnd;

        c->avg_h264_chroma_pixels_tab[1] = ff_avg_h264_chroma_mc4_mmx2;

        c->avg_h264_chroma_pixels_tab[2] = ff_avg_h264_chroma_mc2_mmx2;

        c->put_h264_chroma_pixels_tab[2] = ff_put_h264_chroma_mc2_mmx2;

    }

    if (bit_depth == 10 && CONFIG_H264CHROMA) {

        c->put_h264_chroma_pixels_tab[2] = ff_put_h264_chroma_mc2_10_mmxext;

        c->avg_h264_chroma_pixels_tab[2] = ff_avg_h264_chroma_mc2_10_mmxext;

        c->put_h264_chroma_pixels_tab[1] = ff_put_h264_chroma_mc4_10_mmxext;

        c->avg_h264_chroma_pixels_tab[1] = ff_avg_h264_chroma_mc4_10_mmxext;

    }



    c->add_hfyu_median_prediction   = ff_add_hfyu_median_prediction_mmx2;



    c->scalarproduct_int16          = ff_scalarproduct_int16_mmx2;

    c->scalarproduct_and_madd_int16 = ff_scalarproduct_and_madd_int16_mmx2;



    if (avctx->flags & CODEC_FLAG_BITEXACT) {

        c->apply_window_int16 = ff_apply_window_int16_mmxext_ba;

    } else {

        c->apply_window_int16 = ff_apply_window_int16_mmxext;

    }

#endif

}
