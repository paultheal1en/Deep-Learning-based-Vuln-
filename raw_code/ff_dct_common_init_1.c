av_cold int ff_dct_common_init(MpegEncContext *s)

{

    ff_dsputil_init(&s->dsp, s->avctx);

    ff_videodsp_init(&s->vdsp, 8);



    s->dct_unquantize_h263_intra = dct_unquantize_h263_intra_c;

    s->dct_unquantize_h263_inter = dct_unquantize_h263_inter_c;

    s->dct_unquantize_mpeg1_intra = dct_unquantize_mpeg1_intra_c;

    s->dct_unquantize_mpeg1_inter = dct_unquantize_mpeg1_inter_c;

    s->dct_unquantize_mpeg2_intra = dct_unquantize_mpeg2_intra_c;

    if (s->flags & CODEC_FLAG_BITEXACT)

        s->dct_unquantize_mpeg2_intra = dct_unquantize_mpeg2_intra_bitexact;

    s->dct_unquantize_mpeg2_inter = dct_unquantize_mpeg2_inter_c;



#if ARCH_X86

    ff_MPV_common_init_x86(s);

#elif ARCH_ALPHA

    ff_MPV_common_init_axp(s);

#elif ARCH_ARM

    ff_MPV_common_init_arm(s);

#elif HAVE_ALTIVEC

    ff_MPV_common_init_altivec(s);

#elif ARCH_BFIN

    ff_MPV_common_init_bfin(s);

#endif



    /* load & permutate scantables

     * note: only wmv uses different ones

     */

    if (s->alternate_scan) {

        ff_init_scantable(s->dsp.idct_permutation, &s->inter_scantable  , ff_alternate_vertical_scan);

        ff_init_scantable(s->dsp.idct_permutation, &s->intra_scantable  , ff_alternate_vertical_scan);

    } else {

        ff_init_scantable(s->dsp.idct_permutation, &s->inter_scantable  , ff_zigzag_direct);

        ff_init_scantable(s->dsp.idct_permutation, &s->intra_scantable  , ff_zigzag_direct);

    }

    ff_init_scantable(s->dsp.idct_permutation, &s->intra_h_scantable, ff_alternate_horizontal_scan);

    ff_init_scantable(s->dsp.idct_permutation, &s->intra_v_scantable, ff_alternate_vertical_scan);



    return 0;

}
