void dsputil_init_arm(DSPContext* c, AVCodecContext *avctx)

{

    int idct_algo= avctx->idct_algo;



    ff_put_pixels_clamped = c->put_pixels_clamped;

    ff_add_pixels_clamped = c->add_pixels_clamped;



    if (avctx->lowres == 0) {

        if(idct_algo == FF_IDCT_AUTO){

#if   HAVE_IPP

            idct_algo = FF_IDCT_IPP;

#elif HAVE_NEON

            idct_algo = FF_IDCT_SIMPLENEON;

#elif HAVE_ARMV6

            idct_algo = FF_IDCT_SIMPLEARMV6;

#elif HAVE_ARMV5TE

            idct_algo = FF_IDCT_SIMPLEARMV5TE;

#else

            idct_algo = FF_IDCT_ARM;

#endif

        }



        if(idct_algo==FF_IDCT_ARM){

            c->idct_put= j_rev_dct_ARM_put;

            c->idct_add= j_rev_dct_ARM_add;

            c->idct    = j_rev_dct_ARM;

            c->idct_permutation_type= FF_LIBMPEG2_IDCT_PERM;

        } else if (idct_algo==FF_IDCT_SIMPLEARM){

            c->idct_put= simple_idct_ARM_put;

            c->idct_add= simple_idct_ARM_add;

            c->idct    = simple_idct_ARM;

            c->idct_permutation_type= FF_NO_IDCT_PERM;

#if HAVE_ARMV6

        } else if (idct_algo==FF_IDCT_SIMPLEARMV6){

            c->idct_put= ff_simple_idct_put_armv6;

            c->idct_add= ff_simple_idct_add_armv6;

            c->idct    = ff_simple_idct_armv6;

            c->idct_permutation_type= FF_LIBMPEG2_IDCT_PERM;

#endif

#if HAVE_ARMV5TE

        } else if (idct_algo==FF_IDCT_SIMPLEARMV5TE){

            c->idct_put= simple_idct_put_armv5te;

            c->idct_add= simple_idct_add_armv5te;

            c->idct    = simple_idct_armv5te;

            c->idct_permutation_type = FF_NO_IDCT_PERM;

#endif

#if HAVE_IPP

        } else if (idct_algo==FF_IDCT_IPP){

            c->idct_put= simple_idct_ipp_put;

            c->idct_add= simple_idct_ipp_add;

            c->idct    = simple_idct_ipp;

            c->idct_permutation_type= FF_NO_IDCT_PERM;

#endif

#if HAVE_NEON

        } else if (idct_algo==FF_IDCT_SIMPLENEON){

            c->idct_put= ff_simple_idct_put_neon;

            c->idct_add= ff_simple_idct_add_neon;

            c->idct    = ff_simple_idct_neon;

            c->idct_permutation_type = FF_PARTTRANS_IDCT_PERM;

        } else if ((CONFIG_VP3_DECODER || CONFIG_VP5_DECODER || CONFIG_VP6_DECODER || CONFIG_THEORA_DECODER) &&

                   idct_algo==FF_IDCT_VP3){

            c->idct_put= ff_vp3_idct_put_neon;

            c->idct_add= ff_vp3_idct_add_neon;

            c->idct    = ff_vp3_idct_neon;

            c->idct_permutation_type = FF_TRANSPOSE_IDCT_PERM;

#endif

        }

    }



    c->put_pixels_tab[0][0] = put_pixels16_arm;

    c->put_pixels_tab[0][1] = put_pixels16_x2_arm;

    c->put_pixels_tab[0][2] = put_pixels16_y2_arm;

    c->put_pixels_tab[0][3] = put_pixels16_xy2_arm;

    c->put_no_rnd_pixels_tab[0][0] = put_pixels16_arm;

    c->put_no_rnd_pixels_tab[0][1] = put_no_rnd_pixels16_x2_arm;

    c->put_no_rnd_pixels_tab[0][2] = put_no_rnd_pixels16_y2_arm;

    c->put_no_rnd_pixels_tab[0][3] = put_no_rnd_pixels16_xy2_arm;

    c->put_pixels_tab[1][0] = put_pixels8_arm;

    c->put_pixels_tab[1][1] = put_pixels8_x2_arm;

    c->put_pixels_tab[1][2] = put_pixels8_y2_arm;

    c->put_pixels_tab[1][3] = put_pixels8_xy2_arm;

    c->put_no_rnd_pixels_tab[1][0] = put_pixels8_arm;

    c->put_no_rnd_pixels_tab[1][1] = put_no_rnd_pixels8_x2_arm;

    c->put_no_rnd_pixels_tab[1][2] = put_no_rnd_pixels8_y2_arm;

    c->put_no_rnd_pixels_tab[1][3] = put_no_rnd_pixels8_xy2_arm;



#if HAVE_ARMV5TE

    c->prefetch = ff_prefetch_arm;

#endif



#if HAVE_IWMMXT

    dsputil_init_iwmmxt(c, avctx);

#endif

#if HAVE_ARMVFP

    ff_float_init_arm_vfp(c, avctx);

#endif

#if HAVE_NEON

    ff_dsputil_init_neon(c, avctx);

#endif

}
