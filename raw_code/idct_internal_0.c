static av_always_inline void idct_internal(uint8_t *dst, DCTELEM *block, int stride, int block_stride, int shift, int add){

    int i;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    block[0] += 1<<(shift-1);



    for(i=0; i<4; i++){

        const int z0=  block[0 + block_stride*i]     +  block[2 + block_stride*i];

        const int z1=  block[0 + block_stride*i]     -  block[2 + block_stride*i];

        const int z2= (block[1 + block_stride*i]>>1) -  block[3 + block_stride*i];

        const int z3=  block[1 + block_stride*i]     + (block[3 + block_stride*i]>>1);



        block[0 + block_stride*i]= z0 + z3;

        block[1 + block_stride*i]= z1 + z2;

        block[2 + block_stride*i]= z1 - z2;

        block[3 + block_stride*i]= z0 - z3;

    }



    for(i=0; i<4; i++){

        const int z0=  block[i + block_stride*0]     +  block[i + block_stride*2];

        const int z1=  block[i + block_stride*0]     -  block[i + block_stride*2];

        const int z2= (block[i + block_stride*1]>>1) -  block[i + block_stride*3];

        const int z3=  block[i + block_stride*1]     + (block[i + block_stride*3]>>1);



        dst[i + 0*stride]= cm[ add*dst[i + 0*stride] + ((z0 + z3) >> shift) ];

        dst[i + 1*stride]= cm[ add*dst[i + 1*stride] + ((z1 + z2) >> shift) ];

        dst[i + 2*stride]= cm[ add*dst[i + 2*stride] + ((z1 - z2) >> shift) ];

        dst[i + 3*stride]= cm[ add*dst[i + 3*stride] + ((z0 - z3) >> shift) ];

    }

}
