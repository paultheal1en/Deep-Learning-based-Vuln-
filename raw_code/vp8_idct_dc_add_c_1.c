static void vp8_idct_dc_add_c(uint8_t *dst, DCTELEM block[16], ptrdiff_t stride)

{

    int i, dc = (block[0] + 4) >> 3;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP + dc;

    block[0] = 0;



    for (i = 0; i < 4; i++) {

        dst[0] = cm[dst[0]];

        dst[1] = cm[dst[1]];

        dst[2] = cm[dst[2]];

        dst[3] = cm[dst[3]];

        dst += stride;

    }

}
