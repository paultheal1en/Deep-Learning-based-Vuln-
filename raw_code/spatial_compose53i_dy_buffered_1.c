static void spatial_compose53i_dy_buffered(dwt_compose_t *cs, slice_buffer * sb, int width, int height, int stride_line){

    int y= cs->y;

    int mirror0 = mirror(y-1, height-1);

    int mirror1 = mirror(y  , height-1);

    int mirror2 = mirror(y+1, height-1);

    int mirror3 = mirror(y+2, height-1);



    DWTELEM *b0= cs->b0;

    DWTELEM *b1= cs->b1;

    DWTELEM *b2= slice_buffer_get_line(sb, mirror2 * stride_line);

    DWTELEM *b3= slice_buffer_get_line(sb, mirror3 * stride_line);



{START_TIMER

        if(mirror1 <= mirror3) vertical_compose53iL0(b1, b2, b3, width);

        if(mirror0 <= mirror2) vertical_compose53iH0(b0, b1, b2, width);

STOP_TIMER("vertical_compose53i*")}



{START_TIMER

        if(y-1 >= 0) horizontal_compose53i(b0, width);

        if(mirror0 <= mirror2) horizontal_compose53i(b1, width);

STOP_TIMER("horizontal_compose53i")}



    cs->b0 = b2;

    cs->b1 = b3;

    cs->y += 2;

}
