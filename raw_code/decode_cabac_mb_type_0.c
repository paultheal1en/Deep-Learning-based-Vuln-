static int decode_cabac_mb_type( H264Context *h ) {

    MpegEncContext * const s = &h->s;



    if( h->slice_type == I_TYPE ) {

        return decode_cabac_intra_mb_type(h, 3, 1);

    } else if( h->slice_type == P_TYPE ) {

        if( get_cabac( &h->cabac, &h->cabac_state[14] ) == 0 ) {

            /* P-type */

            if( get_cabac( &h->cabac, &h->cabac_state[15] ) == 0 ) {

                /* P_L0_D16x16, P_8x8 */

                return 3 * get_cabac( &h->cabac, &h->cabac_state[16] );

            } else {

                /* P_L0_D8x16, P_L0_D16x8 */

                return 2 - get_cabac( &h->cabac, &h->cabac_state[17] );

            }

        } else {

            return decode_cabac_intra_mb_type(h, 17, 0) + 5;

        }

    } else if( h->slice_type == B_TYPE ) {

        const int mba_xy = h->left_mb_xy[0];

        const int mbb_xy = h->top_mb_xy;

        int ctx = 0;

        int bits;



        if( h->slice_table[mba_xy] == h->slice_num && !IS_DIRECT( s->current_picture.mb_type[mba_xy] ) )

            ctx++;

        if( h->slice_table[mbb_xy] == h->slice_num && !IS_DIRECT( s->current_picture.mb_type[mbb_xy] ) )

            ctx++;



        if( !get_cabac( &h->cabac, &h->cabac_state[27+ctx] ) )

            return 0; /* B_Direct_16x16 */



        if( !get_cabac( &h->cabac, &h->cabac_state[27+3] ) ) {

            return 1 + get_cabac( &h->cabac, &h->cabac_state[27+5] ); /* B_L[01]_16x16 */

        }



        bits = get_cabac( &h->cabac, &h->cabac_state[27+4] ) << 3;

        bits|= get_cabac( &h->cabac, &h->cabac_state[27+5] ) << 2;

        bits|= get_cabac( &h->cabac, &h->cabac_state[27+5] ) << 1;

        bits|= get_cabac( &h->cabac, &h->cabac_state[27+5] );

        if( bits < 8 )

            return bits + 3; /* B_Bi_16x16 through B_L1_L0_16x8 */

        else if( bits == 13 ) {

            return decode_cabac_intra_mb_type(h, 32, 0) + 23;

        } else if( bits == 14 )

            return 11; /* B_L1_L0_8x16 */

        else if( bits == 15 )

            return 22; /* B_8x8 */



        bits= ( bits<<1 ) | get_cabac( &h->cabac, &h->cabac_state[27+5] );

        return bits - 4; /* B_L0_Bi_* through B_Bi_Bi_* */

    } else {

        /* TODO SI/SP frames? */

        return -1;

    }

}
