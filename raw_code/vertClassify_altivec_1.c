static inline int vertClassify_altivec(uint8_t src[], int stride, PPContext *c) {

    /*

    this code makes no assumption on src or stride.

    One could remove the recomputation of the perm

    vector by assuming (stride % 16) == 0, unfortunately

    this is not always true.

    */

    DECLARE_ALIGNED(16, short, data)[8] =

                    {

                        ((c->nonBQP*c->ppMode.baseDcDiff)>>8) + 1,

                        data[0] * 2 + 1,

                        c->QP * 2,

                        c->QP * 4

                    };

    int numEq;

    uint8_t *src2 = src;

    vector signed short v_dcOffset;

    vector signed short v2QP;

    vector unsigned short v4QP;

    vector unsigned short v_dcThreshold;

    const int properStride = (stride % 16);

    const int srcAlign = ((unsigned long)src2 % 16);

    const int two_vectors = ((srcAlign > 8) || properStride) ? 1 : 0;

    const vector signed int zero = vec_splat_s32(0);

    const vector signed short mask = vec_splat_s16(1);

    vector signed int v_numEq = vec_splat_s32(0);

    vector signed short v_data = vec_ld(0, data);

    vector signed short v_srcAss0, v_srcAss1, v_srcAss2, v_srcAss3,

                        v_srcAss4, v_srcAss5, v_srcAss6, v_srcAss7;

//FIXME avoid this mess if possible

    register int j0 = 0,

                 j1 = stride,

                 j2 = 2 * stride,

                 j3 = 3 * stride,

                 j4 = 4 * stride,

                 j5 = 5 * stride,

                 j6 = 6 * stride,

                 j7 = 7 * stride;

    vector unsigned char v_srcA0, v_srcA1, v_srcA2, v_srcA3,

                         v_srcA4, v_srcA5, v_srcA6, v_srcA7;



    v_dcOffset = vec_splat(v_data, 0);

    v_dcThreshold = (vector unsigned short)vec_splat(v_data, 1);

    v2QP = vec_splat(v_data, 2);

    v4QP = (vector unsigned short)vec_splat(v_data, 3);



    src2 += stride * 4;



#define LOAD_LINE(i)                                                    \

    {                                                                   \

    vector unsigned char perm##i = vec_lvsl(j##i, src2);                \

    vector unsigned char v_srcA2##i;                                    \

    vector unsigned char v_srcA1##i = vec_ld(j##i, src2);               \

    if (two_vectors)                                                    \

        v_srcA2##i = vec_ld(j##i + 16, src2);                           \

    v_srcA##i =                                                         \

        vec_perm(v_srcA1##i, v_srcA2##i, perm##i);                      \

    v_srcAss##i =                                                       \

        (vector signed short)vec_mergeh((vector signed char)zero,       \

                                        (vector signed char)v_srcA##i); }



#define LOAD_LINE_ALIGNED(i)                                            \

    v_srcA##i = vec_ld(j##i, src2);                                     \

    v_srcAss##i =                                                       \

        (vector signed short)vec_mergeh((vector signed char)zero,       \

                                        (vector signed char)v_srcA##i)



    /* Special-casing the aligned case is worthwhile, as all calls from

     * the (transposed) horizontable deblocks will be aligned, in addition

     * to the naturally aligned vertical deblocks. */

    if (properStride && srcAlign) {

        LOAD_LINE_ALIGNED(0);

        LOAD_LINE_ALIGNED(1);

        LOAD_LINE_ALIGNED(2);

        LOAD_LINE_ALIGNED(3);

        LOAD_LINE_ALIGNED(4);

        LOAD_LINE_ALIGNED(5);

        LOAD_LINE_ALIGNED(6);

        LOAD_LINE_ALIGNED(7);

    } else {

        LOAD_LINE(0);

        LOAD_LINE(1);

        LOAD_LINE(2);

        LOAD_LINE(3);

        LOAD_LINE(4);

        LOAD_LINE(5);

        LOAD_LINE(6);

        LOAD_LINE(7);

    }

#undef LOAD_LINE

#undef LOAD_LINE_ALIGNED



#define ITER(i, j)                                                      \

    const vector signed short v_diff##i =                               \

        vec_sub(v_srcAss##i, v_srcAss##j);                              \

    const vector signed short v_sum##i =                                \

        vec_add(v_diff##i, v_dcOffset);                                 \

    const vector signed short v_comp##i =                               \

        (vector signed short)vec_cmplt((vector unsigned short)v_sum##i, \

                                       v_dcThreshold);                  \

    const vector signed short v_part##i = vec_and(mask, v_comp##i);



    {

        ITER(0, 1)

        ITER(1, 2)

        ITER(2, 3)

        ITER(3, 4)

        ITER(4, 5)

        ITER(5, 6)

        ITER(6, 7)



        v_numEq = vec_sum4s(v_part0, v_numEq);

        v_numEq = vec_sum4s(v_part1, v_numEq);

        v_numEq = vec_sum4s(v_part2, v_numEq);

        v_numEq = vec_sum4s(v_part3, v_numEq);

        v_numEq = vec_sum4s(v_part4, v_numEq);

        v_numEq = vec_sum4s(v_part5, v_numEq);

        v_numEq = vec_sum4s(v_part6, v_numEq);

    }



#undef ITER



    v_numEq = vec_sums(v_numEq, zero);



    v_numEq = vec_splat(v_numEq, 3);

    vec_ste(v_numEq, 0, &numEq);



    if (numEq > c->ppMode.flatnessThreshold){

        const vector unsigned char mmoP1 = (const vector unsigned char)

            {0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f,

             0x00, 0x01, 0x12, 0x13, 0x08, 0x09, 0x1A, 0x1B};

        const vector unsigned char mmoP2 = (const vector unsigned char)

            {0x04, 0x05, 0x16, 0x17, 0x0C, 0x0D, 0x1E, 0x1F,

             0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f};

        const vector unsigned char mmoP = (const vector unsigned char)

            vec_lvsl(8, (unsigned char*)0);



        vector signed short mmoL1 = vec_perm(v_srcAss0, v_srcAss2, mmoP1);

        vector signed short mmoL2 = vec_perm(v_srcAss4, v_srcAss6, mmoP2);

        vector signed short mmoL = vec_perm(mmoL1, mmoL2, mmoP);

        vector signed short mmoR1 = vec_perm(v_srcAss5, v_srcAss7, mmoP1);

        vector signed short mmoR2 = vec_perm(v_srcAss1, v_srcAss3, mmoP2);

        vector signed short mmoR = vec_perm(mmoR1, mmoR2, mmoP);

        vector signed short mmoDiff = vec_sub(mmoL, mmoR);

        vector unsigned short mmoSum = (vector unsigned short)vec_add(mmoDiff, v2QP);



        if (vec_any_gt(mmoSum, v4QP))

            return 0;

        else

            return 1;

    }

    else return 2;

}
