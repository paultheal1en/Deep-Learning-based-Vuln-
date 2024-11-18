static void copy_block(uint16_t *pdest, uint16_t *psrc, int block_size, int pitch)

{

    int y;



    for (y = 0; y != block_size; y++, pdest += pitch, psrc += pitch)

        memcpy(pdest, psrc, block_size * sizeof(pdest[0]));

}
