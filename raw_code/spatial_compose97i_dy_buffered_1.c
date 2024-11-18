static void spatial_compose97i_dy_buffered(dwt_compose_t *cs, slice_buffer * sb, int width, int height, int stride_line){

    int y = cs->y;



    int mirror0 = mirror(y - 1, height - 1);

    int mirror1 = mirror(y + 0, height - 1);

    int mirror2 = mirror(y + 1, height - 1);

    int mirror3 = mirror(y + 2, height - 1);

    int mirror4 = mirror(y + 3, height - 1);

    int mirror5 = mirror(y + 4, height - 1);

    DWTELEM *b0= cs->b0;

    DWTELEM *b1= cs->b1;

    DWTELEM *b2= cs->b2;

    DWTELEM *b3= cs->b3;

    DWTELEM *b4= slice_buffer_get_line(sb, mirror4 * stride_line);

    DWTELEM *b5= slice_buffer_get_line(sb, mirror5 * stride_line);



{START_TIMER

    if(y>0 && y+4<height){

        vertical_compose97i(b0, b1, b2, b3, b4, b5, width);

    }else{

        if(mirror3 <= mirror5) vertical_compose97iL1(b3, b4, b5, width);

        if(mirror2 <= mirror4) vertical_compose97iH1(b2, b3, b4, width);

        if(mirror1 <= mirror3) vertical_compose97iL0(b1, b2, b3, width);

        if(mirror0 <= mirror2) vertical_compose97iH0(b0, b1, b2, width);

    }

if(width>400){

STOP_TIMER("vertical_compose97i")}}



{START_TIMER

        if(y-1>=  0) horizontal_compose97i(b0, width);

        if(mirror0 <= mirror2) horizontal_compose97i(b1, width);

if(width>400 && mirror0 <= mirror2){

STOP_TIMER("horizontal_compose97i")}}



    cs->b0=b2;

    cs->b1=b3;

    cs->b2=b4;

    cs->b3=b5;

    cs->y += 2;

}
