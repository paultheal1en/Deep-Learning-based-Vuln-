static void FUNCC(pred4x4_horizontal_add)(uint8_t *_pix, const int16_t *_block,

                                          ptrdiff_t stride)

{

    int i;

    pixel *pix = (pixel*)_pix;

    const dctcoef *block = (const dctcoef*)_block;

    stride >>= sizeof(pixel)-1;

    for(i=0; i<4; i++){

        pixel v = pix[-1];

        pix[0]= v += block[0];

        pix[1]= v += block[1];

        pix[2]= v += block[2];

        pix[3]= v +  block[3];

        pix+= stride;

        block+= 4;

    }

}
