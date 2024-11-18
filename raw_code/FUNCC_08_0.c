static void FUNCC(pred4x4_vertical_add)(uint8_t *_pix, const int16_t *_block,

                                        ptrdiff_t stride)

{

    int i;

    pixel *pix = (pixel*)_pix;

    const dctcoef *block = (const dctcoef*)_block;

    stride >>= sizeof(pixel)-1;

    pix -= stride;

    for(i=0; i<4; i++){

        pixel v = pix[0];

        pix[1*stride]= v += block[0];

        pix[2*stride]= v += block[4];

        pix[3*stride]= v += block[8];

        pix[4*stride]= v +  block[12];

        pix++;

        block++;

    }

}
