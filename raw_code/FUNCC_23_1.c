static void FUNCC(pred8x8_left_dc)(uint8_t *_src, int stride){

    int i;

    int dc0, dc2;

    pixel4 dc0splat, dc2splat;

    pixel *src = (pixel*)_src;

    stride /= sizeof(pixel);



    dc0=dc2=0;

    for(i=0;i<4; i++){

        dc0+= src[-1+i*stride];

        dc2+= src[-1+(i+4)*stride];

    }

    dc0splat = PIXEL_SPLAT_X4((dc0 + 2)>>2);

    dc2splat = PIXEL_SPLAT_X4((dc2 + 2)>>2);



    for(i=0; i<4; i++){

        ((pixel4*)(src+i*stride))[0]=

        ((pixel4*)(src+i*stride))[1]= dc0splat;

    }

    for(i=4; i<8; i++){

        ((pixel4*)(src+i*stride))[0]=

        ((pixel4*)(src+i*stride))[1]= dc2splat;

    }

}
