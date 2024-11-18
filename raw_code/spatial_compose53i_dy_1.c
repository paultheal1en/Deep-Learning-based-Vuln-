static void spatial_compose53i_dy(dwt_compose_t *cs, DWTELEM *buffer, int width, int height, int stride){

    int y= cs->y;

    DWTELEM *b0= cs->b0;

    DWTELEM *b1= cs->b1;

    DWTELEM *b2= buffer + mirror(y+1, height-1)*stride;

    DWTELEM *b3= buffer + mirror(y+2, height-1)*stride;



{START_TIMER

        if(b1 <= b3) vertical_compose53iL0(b1, b2, b3, width);

        if(b0 <= b2) vertical_compose53iH0(b0, b1, b2, width);

STOP_TIMER("vertical_compose53i*")}



{START_TIMER

        if(y-1 >= 0) horizontal_compose53i(b0, width);

        if(b0 <= b2) horizontal_compose53i(b1, width);

STOP_TIMER("horizontal_compose53i")}



    cs->b0 = b2;

    cs->b1 = b3;

    cs->y += 2;

}
