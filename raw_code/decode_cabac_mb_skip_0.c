static int decode_cabac_mb_skip( H264Context *h, int mb_x, int mb_y ) {

    MpegEncContext * const s = &h->s;

    int mba_xy, mbb_xy;

    int ctx = 0;



    if(FRAME_MBAFF){ //FIXME merge with the stuff in fill_caches?

        int mb_xy = mb_x + (mb_y&~1)*s->mb_stride;

        mba_xy = mb_xy - 1;

        if( (mb_y&1)

            && h->slice_table[mba_xy] == h->slice_num

            && MB_FIELD == !!IS_INTERLACED( s->current_picture.mb_type[mba_xy] ) )

            mba_xy += s->mb_stride;

        if( MB_FIELD ){

            mbb_xy = mb_xy - s->mb_stride;

            if( !(mb_y&1)

                && h->slice_table[mbb_xy] == h->slice_num

                && IS_INTERLACED( s->current_picture.mb_type[mbb_xy] ) )

                mbb_xy -= s->mb_stride;

        }else

            mbb_xy = mb_x + (mb_y-1)*s->mb_stride;

    }else{

        int mb_xy = mb_x + mb_y*s->mb_stride;

        mba_xy = mb_xy - 1;

        mbb_xy = mb_xy - s->mb_stride;

    }



    if( h->slice_table[mba_xy] == h->slice_num && !IS_SKIP( s->current_picture.mb_type[mba_xy] ))

        ctx++;

    if( h->slice_table[mbb_xy] == h->slice_num && !IS_SKIP( s->current_picture.mb_type[mbb_xy] ))

        ctx++;



    if( h->slice_type == B_TYPE )

        ctx += 13;

    return get_cabac( &h->cabac, &h->cabac_state[11+ctx] );

}
