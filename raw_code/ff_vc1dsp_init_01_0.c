void ff_vc1dsp_init(DSPContext* dsp, AVCodecContext *avctx) {

    dsp->vc1_inv_trans_8x8 = vc1_inv_trans_8x8_c;

    dsp->vc1_inv_trans_4x8 = vc1_inv_trans_4x8_c;

    dsp->vc1_inv_trans_8x4 = vc1_inv_trans_8x4_c;

    dsp->vc1_inv_trans_4x4 = vc1_inv_trans_4x4_c;

    dsp->vc1_h_overlap = vc1_h_overlap_c;

    dsp->vc1_v_overlap = vc1_v_overlap_c;

    dsp->vc1_loop_filter = vc1_loop_filter;



    dsp->put_vc1_mspel_pixels_tab[ 0] = ff_put_vc1_mspel_mc00_c;

    dsp->put_vc1_mspel_pixels_tab[ 1] = put_vc1_mspel_mc10_c;

    dsp->put_vc1_mspel_pixels_tab[ 2] = put_vc1_mspel_mc20_c;

    dsp->put_vc1_mspel_pixels_tab[ 3] = put_vc1_mspel_mc30_c;

    dsp->put_vc1_mspel_pixels_tab[ 4] = put_vc1_mspel_mc01_c;

    dsp->put_vc1_mspel_pixels_tab[ 5] = put_vc1_mspel_mc11_c;

    dsp->put_vc1_mspel_pixels_tab[ 6] = put_vc1_mspel_mc21_c;

    dsp->put_vc1_mspel_pixels_tab[ 7] = put_vc1_mspel_mc31_c;

    dsp->put_vc1_mspel_pixels_tab[ 8] = put_vc1_mspel_mc02_c;

    dsp->put_vc1_mspel_pixels_tab[ 9] = put_vc1_mspel_mc12_c;

    dsp->put_vc1_mspel_pixels_tab[10] = put_vc1_mspel_mc22_c;

    dsp->put_vc1_mspel_pixels_tab[11] = put_vc1_mspel_mc32_c;

    dsp->put_vc1_mspel_pixels_tab[12] = put_vc1_mspel_mc03_c;

    dsp->put_vc1_mspel_pixels_tab[13] = put_vc1_mspel_mc13_c;

    dsp->put_vc1_mspel_pixels_tab[14] = put_vc1_mspel_mc23_c;

    dsp->put_vc1_mspel_pixels_tab[15] = put_vc1_mspel_mc33_c;



    dsp->avg_vc1_mspel_pixels_tab[ 0] = ff_avg_vc1_mspel_mc00_c;

    dsp->avg_vc1_mspel_pixels_tab[ 1] = avg_vc1_mspel_mc10_c;

    dsp->avg_vc1_mspel_pixels_tab[ 2] = avg_vc1_mspel_mc20_c;

    dsp->avg_vc1_mspel_pixels_tab[ 3] = avg_vc1_mspel_mc30_c;

    dsp->avg_vc1_mspel_pixels_tab[ 4] = avg_vc1_mspel_mc01_c;

    dsp->avg_vc1_mspel_pixels_tab[ 5] = avg_vc1_mspel_mc11_c;

    dsp->avg_vc1_mspel_pixels_tab[ 6] = avg_vc1_mspel_mc21_c;

    dsp->avg_vc1_mspel_pixels_tab[ 7] = avg_vc1_mspel_mc31_c;

    dsp->avg_vc1_mspel_pixels_tab[ 8] = avg_vc1_mspel_mc02_c;

    dsp->avg_vc1_mspel_pixels_tab[ 9] = avg_vc1_mspel_mc12_c;

    dsp->avg_vc1_mspel_pixels_tab[10] = avg_vc1_mspel_mc22_c;

    dsp->avg_vc1_mspel_pixels_tab[11] = avg_vc1_mspel_mc32_c;

    dsp->avg_vc1_mspel_pixels_tab[12] = avg_vc1_mspel_mc03_c;

    dsp->avg_vc1_mspel_pixels_tab[13] = avg_vc1_mspel_mc13_c;

    dsp->avg_vc1_mspel_pixels_tab[14] = avg_vc1_mspel_mc23_c;

    dsp->avg_vc1_mspel_pixels_tab[15] = avg_vc1_mspel_mc33_c;

}
