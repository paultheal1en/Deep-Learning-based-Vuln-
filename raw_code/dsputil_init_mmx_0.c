static av_cold void dsputil_init_mmx(DSPContext *c, AVCodecContext *avctx,

                                     int mm_flags)

{

    const int high_bit_depth = avctx->bits_per_raw_sample > 8;



#if HAVE_INLINE_ASM

    c->put_pixels_clamped        = ff_put_pixels_clamped_mmx;

    c->put_signed_pixels_clamped = ff_put_signed_pixels_clamped_mmx;

    c->add_pixels_clamped        = ff_add_pixels_clamped_mmx;



    if (!high_bit_depth) {

        c->clear_block  = clear_block_mmx;

        c->clear_blocks = clear_blocks_mmx;

        c->draw_edges   = draw_edges_mmx;



        SET_HPEL_FUNCS(put,        [0], 16, mmx);

        SET_HPEL_FUNCS(put_no_rnd, [0], 16, mmx);

        SET_HPEL_FUNCS(avg,        [0], 16, mmx);

        SET_HPEL_FUNCS(avg_no_rnd,    , 16, mmx);

        SET_HPEL_FUNCS(put,        [1],  8, mmx);

        SET_HPEL_FUNCS(put_no_rnd, [1],  8, mmx);

        SET_HPEL_FUNCS(avg,        [1],  8, mmx);



        switch (avctx->idct_algo) {

        case FF_IDCT_AUTO:

        case FF_IDCT_SIMPLEMMX:

            c->idct_put              = ff_simple_idct_put_mmx;

            c->idct_add              = ff_simple_idct_add_mmx;

            c->idct                  = ff_simple_idct_mmx;

            c->idct_permutation_type = FF_SIMPLE_IDCT_PERM;

            break;

        case FF_IDCT_XVIDMMX:

            c->idct_put              = ff_idct_xvid_mmx_put;

            c->idct_add              = ff_idct_xvid_mmx_add;

            c->idct                  = ff_idct_xvid_mmx;

            break;

        }

    }



    c->gmc = gmc_mmx;



    c->add_bytes = add_bytes_mmx;



    if (CONFIG_H263_DECODER || CONFIG_H263_ENCODER) {

        c->h263_v_loop_filter = h263_v_loop_filter_mmx;

        c->h263_h_loop_filter = h263_h_loop_filter_mmx;

    }

#endif /* HAVE_INLINE_ASM */



#if HAVE_YASM

    c->vector_clip_int32 = ff_vector_clip_int32_mmx;

#endif



}
