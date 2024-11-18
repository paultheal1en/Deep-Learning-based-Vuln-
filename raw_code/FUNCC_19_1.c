static void FUNCC(pred8x8l_vertical)(uint8_t *_src, int has_topleft, int has_topright, int _stride)

{

    int y;

    pixel *src = (pixel*)_src;

    int stride = _stride/sizeof(pixel);



    PREDICT_8x8_LOAD_TOP;

    src[0] = t0;

    src[1] = t1;

    src[2] = t2;

    src[3] = t3;

    src[4] = t4;

    src[5] = t5;

    src[6] = t6;

    src[7] = t7;

    for( y = 1; y < 8; y++ ) {

        ((pixel4*)(src+y*stride))[0] = ((pixel4*)src)[0];

        ((pixel4*)(src+y*stride))[1] = ((pixel4*)src)[1];

    }

}
