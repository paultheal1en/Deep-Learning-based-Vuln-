av_cold void ff_rv40dsp_init(RV34DSPContext *c, DSPContext* dsp) {



    ff_rv34dsp_init(c, dsp);



    c->put_pixels_tab[0][ 0] = dsp->put_h264_qpel_pixels_tab[0][0];

    c->put_pixels_tab[0][ 1] = put_rv40_qpel16_mc10_c;

    c->put_pixels_tab[0][ 2] = dsp->put_h264_qpel_pixels_tab[0][2];

    c->put_pixels_tab[0][ 3] = put_rv40_qpel16_mc30_c;

    c->put_pixels_tab[0][ 4] = put_rv40_qpel16_mc01_c;

    c->put_pixels_tab[0][ 5] = put_rv40_qpel16_mc11_c;

    c->put_pixels_tab[0][ 6] = put_rv40_qpel16_mc21_c;

    c->put_pixels_tab[0][ 7] = put_rv40_qpel16_mc31_c;

    c->put_pixels_tab[0][ 8] = dsp->put_h264_qpel_pixels_tab[0][8];

    c->put_pixels_tab[0][ 9] = put_rv40_qpel16_mc12_c;

    c->put_pixels_tab[0][10] = put_rv40_qpel16_mc22_c;

    c->put_pixels_tab[0][11] = put_rv40_qpel16_mc32_c;

    c->put_pixels_tab[0][12] = put_rv40_qpel16_mc03_c;

    c->put_pixels_tab[0][13] = put_rv40_qpel16_mc13_c;

    c->put_pixels_tab[0][14] = put_rv40_qpel16_mc23_c;

    c->put_pixels_tab[0][15] = ff_put_rv40_qpel16_mc33_c;

    c->avg_pixels_tab[0][ 0] = dsp->avg_h264_qpel_pixels_tab[0][0];

    c->avg_pixels_tab[0][ 1] = avg_rv40_qpel16_mc10_c;

    c->avg_pixels_tab[0][ 2] = dsp->avg_h264_qpel_pixels_tab[0][2];

    c->avg_pixels_tab[0][ 3] = avg_rv40_qpel16_mc30_c;

    c->avg_pixels_tab[0][ 4] = avg_rv40_qpel16_mc01_c;

    c->avg_pixels_tab[0][ 5] = avg_rv40_qpel16_mc11_c;

    c->avg_pixels_tab[0][ 6] = avg_rv40_qpel16_mc21_c;

    c->avg_pixels_tab[0][ 7] = avg_rv40_qpel16_mc31_c;

    c->avg_pixels_tab[0][ 8] = dsp->avg_h264_qpel_pixels_tab[0][8];

    c->avg_pixels_tab[0][ 9] = avg_rv40_qpel16_mc12_c;

    c->avg_pixels_tab[0][10] = avg_rv40_qpel16_mc22_c;

    c->avg_pixels_tab[0][11] = avg_rv40_qpel16_mc32_c;

    c->avg_pixels_tab[0][12] = avg_rv40_qpel16_mc03_c;

    c->avg_pixels_tab[0][13] = avg_rv40_qpel16_mc13_c;

    c->avg_pixels_tab[0][14] = avg_rv40_qpel16_mc23_c;

    c->avg_pixels_tab[0][15] = ff_avg_rv40_qpel16_mc33_c;

    c->put_pixels_tab[1][ 0] = dsp->put_h264_qpel_pixels_tab[1][0];

    c->put_pixels_tab[1][ 1] = put_rv40_qpel8_mc10_c;

    c->put_pixels_tab[1][ 2] = dsp->put_h264_qpel_pixels_tab[1][2];

    c->put_pixels_tab[1][ 3] = put_rv40_qpel8_mc30_c;

    c->put_pixels_tab[1][ 4] = put_rv40_qpel8_mc01_c;

    c->put_pixels_tab[1][ 5] = put_rv40_qpel8_mc11_c;

    c->put_pixels_tab[1][ 6] = put_rv40_qpel8_mc21_c;

    c->put_pixels_tab[1][ 7] = put_rv40_qpel8_mc31_c;

    c->put_pixels_tab[1][ 8] = dsp->put_h264_qpel_pixels_tab[1][8];

    c->put_pixels_tab[1][ 9] = put_rv40_qpel8_mc12_c;

    c->put_pixels_tab[1][10] = put_rv40_qpel8_mc22_c;

    c->put_pixels_tab[1][11] = put_rv40_qpel8_mc32_c;

    c->put_pixels_tab[1][12] = put_rv40_qpel8_mc03_c;

    c->put_pixels_tab[1][13] = put_rv40_qpel8_mc13_c;

    c->put_pixels_tab[1][14] = put_rv40_qpel8_mc23_c;

    c->put_pixels_tab[1][15] = ff_put_rv40_qpel8_mc33_c;

    c->avg_pixels_tab[1][ 0] = dsp->avg_h264_qpel_pixels_tab[1][0];

    c->avg_pixels_tab[1][ 1] = avg_rv40_qpel8_mc10_c;

    c->avg_pixels_tab[1][ 2] = dsp->avg_h264_qpel_pixels_tab[1][2];

    c->avg_pixels_tab[1][ 3] = avg_rv40_qpel8_mc30_c;

    c->avg_pixels_tab[1][ 4] = avg_rv40_qpel8_mc01_c;

    c->avg_pixels_tab[1][ 5] = avg_rv40_qpel8_mc11_c;

    c->avg_pixels_tab[1][ 6] = avg_rv40_qpel8_mc21_c;

    c->avg_pixels_tab[1][ 7] = avg_rv40_qpel8_mc31_c;

    c->avg_pixels_tab[1][ 8] = dsp->avg_h264_qpel_pixels_tab[1][8];

    c->avg_pixels_tab[1][ 9] = avg_rv40_qpel8_mc12_c;

    c->avg_pixels_tab[1][10] = avg_rv40_qpel8_mc22_c;

    c->avg_pixels_tab[1][11] = avg_rv40_qpel8_mc32_c;

    c->avg_pixels_tab[1][12] = avg_rv40_qpel8_mc03_c;

    c->avg_pixels_tab[1][13] = avg_rv40_qpel8_mc13_c;

    c->avg_pixels_tab[1][14] = avg_rv40_qpel8_mc23_c;

    c->avg_pixels_tab[1][15] = ff_avg_rv40_qpel8_mc33_c;



    c->put_chroma_pixels_tab[0] = put_rv40_chroma_mc8_c;

    c->put_chroma_pixels_tab[1] = put_rv40_chroma_mc4_c;

    c->avg_chroma_pixels_tab[0] = avg_rv40_chroma_mc8_c;

    c->avg_chroma_pixels_tab[1] = avg_rv40_chroma_mc4_c;



    c->rv40_weight_pixels_tab[0][0] = rv40_weight_func_rnd_16;

    c->rv40_weight_pixels_tab[0][1] = rv40_weight_func_rnd_8;

    c->rv40_weight_pixels_tab[1][0] = rv40_weight_func_nornd_16;

    c->rv40_weight_pixels_tab[1][1] = rv40_weight_func_nornd_8;



    c->rv40_weak_loop_filter[0]     = rv40_h_weak_loop_filter;

    c->rv40_weak_loop_filter[1]     = rv40_v_weak_loop_filter;

    c->rv40_strong_loop_filter[0]   = rv40_h_strong_loop_filter;

    c->rv40_strong_loop_filter[1]   = rv40_v_strong_loop_filter;

    c->rv40_loop_filter_strength[0] = rv40_h_loop_filter_strength;

    c->rv40_loop_filter_strength[1] = rv40_v_loop_filter_strength;



    if (ARCH_X86)

        ff_rv40dsp_init_x86(c, dsp);

    if (HAVE_NEON)

        ff_rv40dsp_init_neon(c, dsp);

}
