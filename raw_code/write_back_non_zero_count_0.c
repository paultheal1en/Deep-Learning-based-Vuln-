static inline void write_back_non_zero_count(H264Context *h){

    MpegEncContext * const s = &h->s;

    const int mb_xy= s->mb_x + s->mb_y*s->mb_stride;

    int n;



    for( n = 0; n < 16+4+4; n++ )

        h->non_zero_count[mb_xy][n] = h->non_zero_count_cache[scan8[n]];

}
