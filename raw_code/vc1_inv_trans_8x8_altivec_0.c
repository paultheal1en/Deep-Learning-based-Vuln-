static void vc1_inv_trans_8x8_altivec(DCTELEM block[64])

{

    vector signed short src0, src1, src2, src3, src4, src5, src6, src7;

    vector signed int s0, s1, s2, s3, s4, s5, s6, s7;

    vector signed int s8, s9, sA, sB, sC, sD, sE, sF;

    vector signed int t0, t1, t2, t3, t4, t5, t6, t7;

    const vector signed int vec_64 = vec_sl(vec_splat_s32(4), vec_splat_u32(4));

    const vector unsigned int vec_7 = vec_splat_u32(7);

    const vector unsigned int vec_4 = vec_splat_u32(4);

    const vector  signed int vec_4s = vec_splat_s32(4);

    const vector unsigned int vec_3 = vec_splat_u32(3);

    const vector unsigned int vec_2 = vec_splat_u32(2);

    const vector  signed int vec_1s = vec_splat_s32(1);

    const vector unsigned int vec_1 = vec_splat_u32(1);





    src0 = vec_ld(  0, block);

    src1 = vec_ld( 16, block);

    src2 = vec_ld( 32, block);

    src3 = vec_ld( 48, block);

    src4 = vec_ld( 64, block);

    src5 = vec_ld( 80, block);

    src6 = vec_ld( 96, block);

    src7 = vec_ld(112, block);



    s0 = vec_unpackl(src0);

    s1 = vec_unpackl(src1);

    s2 = vec_unpackl(src2);

    s3 = vec_unpackl(src3);

    s4 = vec_unpackl(src4);

    s5 = vec_unpackl(src5);

    s6 = vec_unpackl(src6);

    s7 = vec_unpackl(src7);

    s8 = vec_unpackh(src0);

    s9 = vec_unpackh(src1);

    sA = vec_unpackh(src2);

    sB = vec_unpackh(src3);

    sC = vec_unpackh(src4);

    sD = vec_unpackh(src5);

    sE = vec_unpackh(src6);

    sF = vec_unpackh(src7);

    STEP8(s0, s1, s2, s3, s4, s5, s6, s7, vec_4s);

    SHIFT_HOR8(s0, s1, s2, s3, s4, s5, s6, s7);

    STEP8(s8, s9, sA, sB, sC, sD, sE, sF, vec_4s);

    SHIFT_HOR8(s8, s9, sA, sB, sC, sD, sE, sF);

    src0 = vec_pack(s8, s0);

    src1 = vec_pack(s9, s1);

    src2 = vec_pack(sA, s2);

    src3 = vec_pack(sB, s3);

    src4 = vec_pack(sC, s4);

    src5 = vec_pack(sD, s5);

    src6 = vec_pack(sE, s6);

    src7 = vec_pack(sF, s7);

    TRANSPOSE8(src0, src1, src2, src3, src4, src5, src6, src7);



    s0 = vec_unpackl(src0);

    s1 = vec_unpackl(src1);

    s2 = vec_unpackl(src2);

    s3 = vec_unpackl(src3);

    s4 = vec_unpackl(src4);

    s5 = vec_unpackl(src5);

    s6 = vec_unpackl(src6);

    s7 = vec_unpackl(src7);

    s8 = vec_unpackh(src0);

    s9 = vec_unpackh(src1);

    sA = vec_unpackh(src2);

    sB = vec_unpackh(src3);

    sC = vec_unpackh(src4);

    sD = vec_unpackh(src5);

    sE = vec_unpackh(src6);

    sF = vec_unpackh(src7);

    STEP8(s0, s1, s2, s3, s4, s5, s6, s7, vec_64);

    SHIFT_VERT8(s0, s1, s2, s3, s4, s5, s6, s7);

    STEP8(s8, s9, sA, sB, sC, sD, sE, sF, vec_64);

    SHIFT_VERT8(s8, s9, sA, sB, sC, sD, sE, sF);

    src0 = vec_pack(s8, s0);

    src1 = vec_pack(s9, s1);

    src2 = vec_pack(sA, s2);

    src3 = vec_pack(sB, s3);

    src4 = vec_pack(sC, s4);

    src5 = vec_pack(sD, s5);

    src6 = vec_pack(sE, s6);

    src7 = vec_pack(sF, s7);



    vec_st(src0,  0, block);

    vec_st(src1, 16, block);

    vec_st(src2, 32, block);

    vec_st(src3, 48, block);

    vec_st(src4, 64, block);

    vec_st(src5, 80, block);

    vec_st(src6, 96, block);

    vec_st(src7,112, block);

}
