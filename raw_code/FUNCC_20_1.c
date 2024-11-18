static void FUNCC(pred8x8_vertical)(uint8_t *_src, int _stride){

    int i;

    pixel *src = (pixel*)_src;

    int stride = _stride/sizeof(pixel);

    const pixel4 a= ((pixel4*)(src-stride))[0];

    const pixel4 b= ((pixel4*)(src-stride))[1];



    for(i=0; i<8; i++){

        ((pixel4*)(src+i*stride))[0]= a;

        ((pixel4*)(src+i*stride))[1]= b;

    }

}
