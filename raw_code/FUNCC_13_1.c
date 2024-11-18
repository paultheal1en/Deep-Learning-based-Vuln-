void FUNCC(ff_h264_idct8_dc_add)(uint8_t *_dst, DCTELEM *block, int stride){

    int i, j;

    int dc = (((dctcoef*)block)[0] + 32) >> 6;

    INIT_CLIP

    pixel *dst = (pixel*)_dst;

    stride /= sizeof(pixel);

    for( j = 0; j < 8; j++ )

    {

        for( i = 0; i < 8; i++ )

            dst[i] = CLIP( dst[i] + dc );

        dst += stride;

    }

}
