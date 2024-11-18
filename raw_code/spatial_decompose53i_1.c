static void spatial_decompose53i(DWTELEM *buffer, int width, int height, int stride){

    int y;

    DWTELEM *b0= buffer + mirror(-2-1, height-1)*stride;

    DWTELEM *b1= buffer + mirror(-2  , height-1)*stride;



    for(y=-2; y<height; y+=2){

        DWTELEM *b2= buffer + mirror(y+1, height-1)*stride;

        DWTELEM *b3= buffer + mirror(y+2, height-1)*stride;



{START_TIMER

        if(b1 <= b3)     horizontal_decompose53i(b2, width);

        if(y+2 < height) horizontal_decompose53i(b3, width);

STOP_TIMER("horizontal_decompose53i")}



{START_TIMER

        if(b1 <= b3) vertical_decompose53iH0(b1, b2, b3, width);

        if(b0 <= b2) vertical_decompose53iL0(b0, b1, b2, width);

STOP_TIMER("vertical_decompose53i*")}



        b0=b2;

        b1=b3;

    }

}
