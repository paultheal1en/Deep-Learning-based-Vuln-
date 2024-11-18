static void rv34_idct_add_c(uint8_t *dst, ptrdiff_t stride, DCTELEM *block){

    int      temp[16];

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    int      i;



    rv34_row_transform(temp, block);

    memset(block, 0, 16*sizeof(DCTELEM));



    for(i = 0; i < 4; i++){

        const int z0 = 13*(temp[4*0+i] +    temp[4*2+i]) + 0x200;

        const int z1 = 13*(temp[4*0+i] -    temp[4*2+i]) + 0x200;

        const int z2 =  7* temp[4*1+i] - 17*temp[4*3+i];

        const int z3 = 17* temp[4*1+i] +  7*temp[4*3+i];



        dst[0] = cm[ dst[0] + ( (z0 + z3) >> 10 ) ];

        dst[1] = cm[ dst[1] + ( (z1 + z2) >> 10 ) ];

        dst[2] = cm[ dst[2] + ( (z1 - z2) >> 10 ) ];

        dst[3] = cm[ dst[3] + ( (z0 - z3) >> 10 ) ];



        dst  += stride;

    }

}
