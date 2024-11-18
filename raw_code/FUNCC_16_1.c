static void FUNCC(pred4x4_vertical)(uint8_t *_src, const uint8_t *topright, int _stride){

    pixel *src = (pixel*)_src;

    int stride = _stride/sizeof(pixel);

    const pixel4 a= ((pixel4*)(src-stride))[0];

    ((pixel4*)(src+0*stride))[0]= a;

    ((pixel4*)(src+1*stride))[0]= a;

    ((pixel4*)(src+2*stride))[0]= a;

    ((pixel4*)(src+3*stride))[0]= a;

}
