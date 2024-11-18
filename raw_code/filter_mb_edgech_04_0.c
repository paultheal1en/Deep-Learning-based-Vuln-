static void filter_mb_edgech( H264Context *h, uint8_t *pix, int stride, int bS[4], int qp ) {

    int i, d;

    const int index_a = clip( qp + h->slice_alpha_c0_offset, 0, 51 );

    const int alpha = alpha_table[index_a];

    const int beta  = beta_table[clip( qp + h->slice_beta_offset, 0, 51 )];

    const int pix_next  = stride;



    for( i = 0; i < 4; i++ )

    {

        if( bS[i] == 0 ) {

            pix += 2;

            continue;

        }



        /* 2px edge length (see deblocking_filter_edgecv) */

        for( d = 0; d < 2; d++ ) {

            const uint8_t p0 = pix[-1*pix_next];

            const uint8_t p1 = pix[-2*pix_next];

            const uint8_t q0 = pix[0];

            const uint8_t q1 = pix[1*pix_next];



            if( abs( p0 - q0 ) >= alpha ||

                abs( p1 - p0 ) >= beta ||

                abs( q1 - q0 ) >= beta ) {

                pix++;

                continue;

            }



            if( bS[i] < 4 ) {

                int tc = tc0_table[index_a][bS[i] - 1] + 1;

                int i_delta = clip( (((q0 - p0 ) << 2) + (p1 - q1) + 4) >> 3, -tc, tc );



                pix[-pix_next] = clip( p0 + i_delta, 0, 255 );    /* p0' */

                pix[0]         = clip( q0 - i_delta, 0, 255 );    /* q0' */

            }

            else

            {

                pix[-pix_next] = ( 2*p1 + p0 + q1 + 2 ) >> 2;   /* p0' */

                pix[0]         = ( 2*q1 + q0 + p1 + 2 ) >> 2;   /* q0' */

            }

            pix++;

        }

    }

}
