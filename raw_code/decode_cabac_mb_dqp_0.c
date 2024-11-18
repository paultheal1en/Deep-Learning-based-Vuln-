static int decode_cabac_mb_dqp( H264Context *h) {

    MpegEncContext * const s = &h->s;

    int mbn_xy;

    int   ctx = 0;

    int   val = 0;



    if( s->mb_x > 0 )

        mbn_xy = s->mb_x + s->mb_y*s->mb_stride - 1;

    else

        mbn_xy = s->mb_width - 1 + (s->mb_y-1)*s->mb_stride;



    if( h->last_qscale_diff != 0 )

        ctx++;



    while( get_cabac( &h->cabac, &h->cabac_state[60 + ctx] ) ) {

        if( ctx < 2 )

            ctx = 2;

        else

            ctx = 3;

        val++;

        if(val > 102) //prevent infinite loop

            return INT_MIN;

    }



    if( val&0x01 )

        return (val + 1)/2;

    else

        return -(val + 1)/2;

}
