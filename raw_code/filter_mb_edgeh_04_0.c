static void filter_mb_edgeh( H264Context *h, uint8_t *pix, int stride, int16_t bS[4], int qp ) {

    int i, d;

    const int index_a = qp + h->slice_alpha_c0_offset;

    const int alpha = (alpha_table+52)[index_a];

    const int beta  = (beta_table+52)[qp + h->slice_beta_offset];

    const int pix_next  = stride;



    if( bS[0] < 4 ) {

        int8_t tc[4];

        for(i=0; i<4; i++)

            tc[i] = bS[i] ? (tc0_table+52)[index_a][bS[i] - 1] : -1;

        h->s.dsp.h264_v_loop_filter_luma(pix, stride, alpha, beta, tc);

    } else {

        h->s.dsp.h264_v_loop_filter_luma_intra(pix, stride, alpha, beta);

    }

}
