static void ff_jref_idct1_put(uint8_t *dest, int line_size, DCTELEM *block)

{

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    dest[0] = cm[(block[0] + 4)>>3];

}
