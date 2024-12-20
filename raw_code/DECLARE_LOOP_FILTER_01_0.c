DECLARE_LOOP_FILTER(mmxext)

DECLARE_LOOP_FILTER(sse2)

DECLARE_LOOP_FILTER(ssse3)

DECLARE_LOOP_FILTER(sse4)



#endif



#define VP8_LUMA_MC_FUNC(IDX, SIZE, OPT) \

    c->put_vp8_epel_pixels_tab[IDX][0][2] = ff_put_vp8_epel ## SIZE ## _h6_ ## OPT; \

    c->put_vp8_epel_pixels_tab[IDX][2][0] = ff_put_vp8_epel ## SIZE ## _v6_ ## OPT; \

    c->put_vp8_epel_pixels_tab[IDX][2][2] = ff_put_vp8_epel ## SIZE ## _h6v6_ ## OPT



#define VP8_MC_FUNC(IDX, SIZE, OPT) \

    c->put_vp8_epel_pixels_tab[IDX][0][1] = ff_put_vp8_epel ## SIZE ## _h4_ ## OPT; \

    c->put_vp8_epel_pixels_tab[IDX][1][0] = ff_put_vp8_epel ## SIZE ## _v4_ ## OPT; \

    c->put_vp8_epel_pixels_tab[IDX][1][1] = ff_put_vp8_epel ## SIZE ## _h4v4_ ## OPT; \

    c->put_vp8_epel_pixels_tab[IDX][1][2] = ff_put_vp8_epel ## SIZE ## _h6v4_ ## OPT; \

    c->put_vp8_epel_pixels_tab[IDX][2][1] = ff_put_vp8_epel ## SIZE ## _h4v6_ ## OPT; \

    VP8_LUMA_MC_FUNC(IDX, SIZE, OPT)



#define VP8_BILINEAR_MC_FUNC(IDX, SIZE, OPT) \

    c->put_vp8_bilinear_pixels_tab[IDX][0][1] = ff_put_vp8_bilinear ## SIZE ## _h_ ## OPT; \

    c->put_vp8_bilinear_pixels_tab[IDX][0][2] = ff_put_vp8_bilinear ## SIZE ## _h_ ## OPT; \

    c->put_vp8_bilinear_pixels_tab[IDX][1][0] = ff_put_vp8_bilinear ## SIZE ## _v_ ## OPT; \

    c->put_vp8_bilinear_pixels_tab[IDX][1][1] = ff_put_vp8_bilinear ## SIZE ## _hv_ ## OPT; \

    c->put_vp8_bilinear_pixels_tab[IDX][1][2] = ff_put_vp8_bilinear ## SIZE ## _hv_ ## OPT; \

    c->put_vp8_bilinear_pixels_tab[IDX][2][0] = ff_put_vp8_bilinear ## SIZE ## _v_ ## OPT; \

    c->put_vp8_bilinear_pixels_tab[IDX][2][1] = ff_put_vp8_bilinear ## SIZE ## _hv_ ## OPT; \

    c->put_vp8_bilinear_pixels_tab[IDX][2][2] = ff_put_vp8_bilinear ## SIZE ## _hv_ ## OPT





av_cold void ff_vp8dsp_init_x86(VP8DSPContext* c)

{

    mm_flags = mm_support();



#if HAVE_YASM

    if (mm_flags & FF_MM_MMX) {

        c->vp8_idct_dc_add    = ff_vp8_idct_dc_add_mmx;

        c->vp8_idct_dc_add4y  = ff_vp8_idct_dc_add4y_mmx;

        c->vp8_idct_dc_add4uv = ff_vp8_idct_dc_add4uv_mmx;

        c->vp8_idct_add       = ff_vp8_idct_add_mmx;

        c->vp8_luma_dc_wht    = ff_vp8_luma_dc_wht_mmx;

        c->put_vp8_epel_pixels_tab[0][0][0]     =

        c->put_vp8_bilinear_pixels_tab[0][0][0] = ff_put_vp8_pixels16_mmx;

        c->put_vp8_epel_pixels_tab[1][0][0]     =

        c->put_vp8_bilinear_pixels_tab[1][0][0] = ff_put_vp8_pixels8_mmx;



        c->vp8_v_loop_filter_simple = ff_vp8_v_loop_filter_simple_mmx;

        c->vp8_h_loop_filter_simple = ff_vp8_h_loop_filter_simple_mmx;



        c->vp8_v_loop_filter16y_inner = ff_vp8_v_loop_filter16y_inner_mmx;

        c->vp8_h_loop_filter16y_inner = ff_vp8_h_loop_filter16y_inner_mmx;

        c->vp8_v_loop_filter8uv_inner = ff_vp8_v_loop_filter8uv_inner_mmx;

        c->vp8_h_loop_filter8uv_inner = ff_vp8_h_loop_filter8uv_inner_mmx;



        c->vp8_v_loop_filter16y       = ff_vp8_v_loop_filter16y_mbedge_mmx;

        c->vp8_h_loop_filter16y       = ff_vp8_h_loop_filter16y_mbedge_mmx;

        c->vp8_v_loop_filter8uv       = ff_vp8_v_loop_filter8uv_mbedge_mmx;

        c->vp8_h_loop_filter8uv       = ff_vp8_h_loop_filter8uv_mbedge_mmx;

    }



    /* note that 4-tap width=16 functions are missing because w=16

     * is only used for luma, and luma is always a copy or sixtap. */

    if (mm_flags & FF_MM_MMX2) {

        VP8_LUMA_MC_FUNC(0, 16, mmxext);

        VP8_MC_FUNC(1, 8, mmxext);

        VP8_MC_FUNC(2, 4, mmxext);

        VP8_BILINEAR_MC_FUNC(0, 16, mmxext);

        VP8_BILINEAR_MC_FUNC(1, 8, mmxext);

        VP8_BILINEAR_MC_FUNC(2, 4, mmxext);



        c->vp8_v_loop_filter_simple = ff_vp8_v_loop_filter_simple_mmxext;

        c->vp8_h_loop_filter_simple = ff_vp8_h_loop_filter_simple_mmxext;



        c->vp8_v_loop_filter16y_inner = ff_vp8_v_loop_filter16y_inner_mmxext;

        c->vp8_h_loop_filter16y_inner = ff_vp8_h_loop_filter16y_inner_mmxext;

        c->vp8_v_loop_filter8uv_inner = ff_vp8_v_loop_filter8uv_inner_mmxext;

        c->vp8_h_loop_filter8uv_inner = ff_vp8_h_loop_filter8uv_inner_mmxext;



        c->vp8_v_loop_filter16y       = ff_vp8_v_loop_filter16y_mbedge_mmxext;

        c->vp8_h_loop_filter16y       = ff_vp8_h_loop_filter16y_mbedge_mmxext;

        c->vp8_v_loop_filter8uv       = ff_vp8_v_loop_filter8uv_mbedge_mmxext;

        c->vp8_h_loop_filter8uv       = ff_vp8_h_loop_filter8uv_mbedge_mmxext;

    }



    if (mm_flags & FF_MM_SSE) {

        c->vp8_idct_add                         = ff_vp8_idct_add_sse;

        c->put_vp8_epel_pixels_tab[0][0][0]     =

        c->put_vp8_bilinear_pixels_tab[0][0][0] = ff_put_vp8_pixels16_sse;

    }



    if (mm_flags & (FF_MM_SSE2|FF_MM_SSE2SLOW)) {

        VP8_LUMA_MC_FUNC(0, 16, sse2);

        VP8_MC_FUNC(1, 8, sse2);

        VP8_BILINEAR_MC_FUNC(0, 16, sse2);

        VP8_BILINEAR_MC_FUNC(1, 8, sse2);



        c->vp8_v_loop_filter_simple = ff_vp8_v_loop_filter_simple_sse2;

        c->vp8_h_loop_filter_simple = ff_vp8_h_loop_filter_simple_sse2;



        c->vp8_v_loop_filter16y_inner = ff_vp8_v_loop_filter16y_inner_sse2;

        c->vp8_v_loop_filter8uv_inner = ff_vp8_v_loop_filter8uv_inner_sse2;



        c->vp8_v_loop_filter16y       = ff_vp8_v_loop_filter16y_mbedge_sse2;

        c->vp8_v_loop_filter8uv       = ff_vp8_v_loop_filter8uv_mbedge_sse2;

    }



    if (mm_flags & FF_MM_SSE2) {

        c->vp8_idct_dc_add4y          = ff_vp8_idct_dc_add4y_sse2;



        c->vp8_h_loop_filter16y_inner = ff_vp8_h_loop_filter16y_inner_sse2;

        c->vp8_h_loop_filter8uv_inner = ff_vp8_h_loop_filter8uv_inner_sse2;



        c->vp8_h_loop_filter16y       = ff_vp8_h_loop_filter16y_mbedge_sse2;

        c->vp8_h_loop_filter8uv       = ff_vp8_h_loop_filter8uv_mbedge_sse2;

    }



    if (mm_flags & FF_MM_SSSE3) {

        VP8_LUMA_MC_FUNC(0, 16, ssse3);

        VP8_MC_FUNC(1, 8, ssse3);

        VP8_MC_FUNC(2, 4, ssse3);

        VP8_BILINEAR_MC_FUNC(0, 16, ssse3);

        VP8_BILINEAR_MC_FUNC(1, 8, ssse3);

        VP8_BILINEAR_MC_FUNC(2, 4, ssse3);



        c->vp8_v_loop_filter_simple = ff_vp8_v_loop_filter_simple_ssse3;

        c->vp8_h_loop_filter_simple = ff_vp8_h_loop_filter_simple_ssse3;



        c->vp8_v_loop_filter16y_inner = ff_vp8_v_loop_filter16y_inner_ssse3;

        c->vp8_h_loop_filter16y_inner = ff_vp8_h_loop_filter16y_inner_ssse3;

        c->vp8_v_loop_filter8uv_inner = ff_vp8_v_loop_filter8uv_inner_ssse3;

        c->vp8_h_loop_filter8uv_inner = ff_vp8_h_loop_filter8uv_inner_ssse3;



        c->vp8_v_loop_filter16y       = ff_vp8_v_loop_filter16y_mbedge_ssse3;

        c->vp8_h_loop_filter16y       = ff_vp8_h_loop_filter16y_mbedge_ssse3;

        c->vp8_v_loop_filter8uv       = ff_vp8_v_loop_filter8uv_mbedge_ssse3;

        c->vp8_h_loop_filter8uv       = ff_vp8_h_loop_filter8uv_mbedge_ssse3;

    }



    if (mm_flags & FF_MM_SSE4) {

        c->vp8_idct_dc_add                  = ff_vp8_idct_dc_add_sse4;



        c->vp8_h_loop_filter16y       = ff_vp8_h_loop_filter16y_mbedge_sse4;

        c->vp8_h_loop_filter8uv       = ff_vp8_h_loop_filter8uv_mbedge_sse4;

    }

#endif

}
