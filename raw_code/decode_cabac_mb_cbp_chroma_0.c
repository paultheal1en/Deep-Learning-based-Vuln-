static int decode_cabac_mb_cbp_chroma( H264Context *h) {

    int ctx;

    int cbp_a, cbp_b;



    cbp_a = (h->left_cbp>>4)&0x03;

    cbp_b = (h-> top_cbp>>4)&0x03;



    ctx = 0;

    if( cbp_a > 0 ) ctx++;

    if( cbp_b > 0 ) ctx += 2;

    if( get_cabac( &h->cabac, &h->cabac_state[77 + ctx] ) == 0 )

        return 0;



    ctx = 4;

    if( cbp_a == 2 ) ctx++;

    if( cbp_b == 2 ) ctx += 2;

    return 1 + get_cabac( &h->cabac, &h->cabac_state[77 + ctx] );

}
