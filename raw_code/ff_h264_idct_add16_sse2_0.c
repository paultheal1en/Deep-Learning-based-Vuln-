static void ff_h264_idct_add16_sse2(uint8_t *dst, const int *block_offset, DCTELEM *block, int stride, const uint8_t nnzc[6*8]){

    int i;

    for(i=0; i<16; i+=2)

        if(nnzc[ scan8[i+0] ]|nnzc[ scan8[i+1] ])

            ff_x264_add8x4_idct_sse2 (dst + block_offset[i], block + i*16, stride);

}
