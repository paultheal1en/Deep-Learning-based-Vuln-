static void ff_h264_idct8_add4_mmx(uint8_t *dst, const int *block_offset, DCTELEM *block, int stride, const uint8_t nnzc[6*8]){

    int i;

    for(i=0; i<16; i+=4){

        if(nnzc[ scan8[i] ])

            ff_h264_idct8_add_mmx(dst + block_offset[i], block + i*16, stride);

    }

}
