av_cold void ff_h264dsp_init_ppc(H264DSPContext *c, const int bit_depth,

                                 const int chroma_format_idc)

{

#if HAVE_ALTIVEC

    if (!(av_get_cpu_flags() & AV_CPU_FLAG_ALTIVEC))

        return;



    if (bit_depth == 8) {

        c->h264_idct_add = h264_idct_add_altivec;

        if (chroma_format_idc == 1)

            c->h264_idct_add8 = h264_idct_add8_altivec;

        c->h264_idct_add16      = h264_idct_add16_altivec;

        c->h264_idct_add16intra = h264_idct_add16intra_altivec;

        c->h264_idct_dc_add= h264_idct_dc_add_altivec;

        c->h264_idct8_dc_add = h264_idct8_dc_add_altivec;

        c->h264_idct8_add    = h264_idct8_add_altivec;

        c->h264_idct8_add4   = h264_idct8_add4_altivec;

        c->h264_v_loop_filter_luma= h264_v_loop_filter_luma_altivec;

        c->h264_h_loop_filter_luma= h264_h_loop_filter_luma_altivec;



        c->weight_h264_pixels_tab[0]   = weight_h264_pixels16_altivec;

        c->weight_h264_pixels_tab[1]   = weight_h264_pixels8_altivec;

        c->biweight_h264_pixels_tab[0] = biweight_h264_pixels16_altivec;

        c->biweight_h264_pixels_tab[1] = biweight_h264_pixels8_altivec;

    }

#endif /* HAVE_ALTIVEC */

}
