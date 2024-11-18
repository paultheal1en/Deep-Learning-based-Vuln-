static void vc1_inv_trans_4x4_dc_c(uint8_t *dest, int linesize, DCTELEM *block)

{

    int i;

    int dc = block[0];

    const uint8_t *cm;

    dc = (17 * dc +  4) >> 3;

    dc = (17 * dc + 64) >> 7;

    cm = ff_cropTbl + MAX_NEG_CROP + dc;

    for(i = 0; i < 4; i++){

        dest[0] = cm[dest[0]];

        dest[1] = cm[dest[1]];

        dest[2] = cm[dest[2]];

        dest[3] = cm[dest[3]];

        dest += linesize;

    }

}
