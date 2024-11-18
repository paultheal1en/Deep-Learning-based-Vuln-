static int decode_cabac_mb_chroma_pre_mode( H264Context *h) {

    const int mba_xy = h->left_mb_xy[0];

    const int mbb_xy = h->top_mb_xy;



    int ctx = 0;



    /* No need to test for IS_INTRA4x4 and IS_INTRA16x16, as we set chroma_pred_mode_table to 0 */

    if( h->slice_table[mba_xy] == h->slice_num && h->chroma_pred_mode_table[mba_xy] != 0 )

        ctx++;



    if( h->slice_table[mbb_xy] == h->slice_num && h->chroma_pred_mode_table[mbb_xy] != 0 )

        ctx++;



    if( get_cabac( &h->cabac, &h->cabac_state[64+ctx] ) == 0 )

        return 0;



    if( get_cabac( &h->cabac, &h->cabac_state[64+3] ) == 0 )

        return 1;

    if( get_cabac( &h->cabac, &h->cabac_state[64+3] ) == 0 )

        return 2;

    else

        return 3;

}
