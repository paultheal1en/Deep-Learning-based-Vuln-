static void FUNCC(pred16x16_vertical)(uint8_t *_src, int _stride){

    int i;

    pixel *src = (pixel*)_src;

    int stride = _stride/sizeof(pixel);

    const pixel4 a = ((pixel4*)(src-stride))[0];

    const pixel4 b = ((pixel4*)(src-stride))[1];

    const pixel4 c = ((pixel4*)(src-stride))[2];

    const pixel4 d = ((pixel4*)(src-stride))[3];



    for(i=0; i<16; i++){

        ((pixel4*)(src+i*stride))[0] = a;

        ((pixel4*)(src+i*stride))[1] = b;

        ((pixel4*)(src+i*stride))[2] = c;

        ((pixel4*)(src+i*stride))[3] = d;

    }

}
