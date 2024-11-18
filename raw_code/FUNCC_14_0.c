static void FUNCC(pred8x8l_vertical_add)(uint8_t *_pix, const int16_t *_block,

                                         ptrdiff_t stride)

{

    int i;

    pixel *pix = (pixel*)_pix;

    const dctcoef *block = (const dctcoef*)_block;

    stride >>= sizeof(pixel)-1;

    pix -= stride;

    for(i=0; i<8; i++){

        pixel v = pix[0];

        pix[1*stride]= v += block[0];

        pix[2*stride]= v += block[8];

        pix[3*stride]= v += block[16];

        pix[4*stride]= v += block[24];

        pix[5*stride]= v += block[32];

        pix[6*stride]= v += block[40];

        pix[7*stride]= v += block[48];

        pix[8*stride]= v +  block[56];

        pix++;

        block++;

    }

}
