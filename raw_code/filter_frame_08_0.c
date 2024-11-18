static void filter_frame(H264Context *h) {

    int mb_x = 0;

    int mb_y = 0;



    for( mb_y = 0; mb_y < h->s.mb_height; mb_y++ ) {

        for( mb_x = 0; mb_x < h->s.mb_width; mb_x++ ) {

            filter_mb( h, mb_x, mb_y );

        }

    }

}
