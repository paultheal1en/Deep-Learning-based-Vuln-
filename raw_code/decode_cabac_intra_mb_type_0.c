static int decode_cabac_intra_mb_type(H264Context *h, int ctx_base, int intra_slice) {

    uint8_t *state= &h->cabac_state[ctx_base];

    int mb_type;



    if(intra_slice){

        MpegEncContext * const s = &h->s;

        const int mba_xy = h->left_mb_xy[0];

        const int mbb_xy = h->top_mb_xy;

        int ctx=0;

        if( h->slice_table[mba_xy] == h->slice_num && !IS_INTRA4x4( s->current_picture.mb_type[mba_xy] ) )

            ctx++;

        if( h->slice_table[mbb_xy] == h->slice_num && !IS_INTRA4x4( s->current_picture.mb_type[mbb_xy] ) )

            ctx++;

        if( get_cabac_noinline( &h->cabac, &state[ctx] ) == 0 )

            return 0;   /* I4x4 */

        state += 2;

    }else{

        if( get_cabac_noinline( &h->cabac, state ) == 0 )

            return 0;   /* I4x4 */

    }



    if( get_cabac_terminate( &h->cabac ) )

        return 25;  /* PCM */



    mb_type = 1; /* I16x16 */

    mb_type += 12 * get_cabac_noinline( &h->cabac, &state[1] ); /* cbp_luma != 0 */

    if( get_cabac_noinline( &h->cabac, &state[2] ) ) /* cbp_chroma */

        mb_type += 4 + 4 * get_cabac_noinline( &h->cabac, &state[2+intra_slice] );

    mb_type += 2 * get_cabac_noinline( &h->cabac, &state[3+intra_slice] );

    mb_type += 1 * get_cabac_noinline( &h->cabac, &state[3+2*intra_slice] );

    return mb_type;

}
