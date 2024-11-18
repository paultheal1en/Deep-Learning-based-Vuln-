static inline void doVertLowPass_altivec(uint8_t *src, int stride, PPContext *c) {

    /*

    this code makes no assumption on src or stride.

    One could remove the recomputation of the perm

    vector by assuming (stride % 16) == 0, unfortunately

    this is not always true. Quite a lot of load/stores

    can be removed by assuming proper alignment of

    src & stride :-(

    */

    uint8_t *src2 = src;

    const vector signed int zero = vec_splat_s32(0);

    const int properStride = (stride % 16);

    const int srcAlign = ((unsigned long)src2 % 16);

    DECLARE_ALIGNED(16, short, qp[8]) = {c->QP};

    vector signed short vqp = vec_ld(0, qp);

    vector signed short vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7, vb8, vb9;

    vector unsigned char vbA0, vbA1, vbA2, vbA3, vbA4, vbA5, vbA6, vbA7, vbA8, vbA9;

    vector unsigned char vbB0, vbB1, vbB2, vbB3, vbB4, vbB5, vbB6, vbB7, vbB8, vbB9;

    vector unsigned char vbT0, vbT1, vbT2, vbT3, vbT4, vbT5, vbT6, vbT7, vbT8, vbT9;

    vector unsigned char perml0, perml1, perml2, perml3, perml4,

                         perml5, perml6, perml7, perml8, perml9;

    register int j0 = 0,

                 j1 = stride,

                 j2 = 2 * stride,

                 j3 = 3 * stride,

                 j4 = 4 * stride,

                 j5 = 5 * stride,

                 j6 = 6 * stride,

                 j7 = 7 * stride,

                 j8 = 8 * stride,

                 j9 = 9 * stride;



    vqp = vec_splat(vqp, 0);



    src2 += stride*3;



#define LOAD_LINE(i)                                                    \

    perml##i = vec_lvsl(i * stride, src2);                              \

    vbA##i = vec_ld(i * stride, src2);                                  \

    vbB##i = vec_ld(i * stride + 16, src2);                             \

    vbT##i = vec_perm(vbA##i, vbB##i, perml##i);                        \

    vb##i =                                                             \

        (vector signed short)vec_mergeh((vector unsigned char)zero,     \

                                        (vector unsigned char)vbT##i)



#define LOAD_LINE_ALIGNED(i)                                            \

    vbT##i = vec_ld(j##i, src2);                                        \

    vb##i =                                                             \

        (vector signed short)vec_mergeh((vector signed char)zero,       \

                                        (vector signed char)vbT##i)



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

          LOAD_LINE_ALIGNED(8);

          LOAD_LINE_ALIGNED(9);

    } else {

          LOAD_LINE(0);

          LOAD_LINE(1);

          LOAD_LINE(2);

          LOAD_LINE(3);

          LOAD_LINE(4);

          LOAD_LINE(5);

          LOAD_LINE(6);

          LOAD_LINE(7);

          LOAD_LINE(8);

          LOAD_LINE(9);

    }

#undef LOAD_LINE

#undef LOAD_LINE_ALIGNED

    {

        const vector unsigned short v_2 = vec_splat_u16(2);

        const vector unsigned short v_4 = vec_splat_u16(4);



        const vector signed short v_diff01 = vec_sub(vb0, vb1);

        const vector unsigned short v_cmp01 =

            (const vector unsigned short) vec_cmplt(vec_abs(v_diff01), vqp);

        const vector signed short v_first = vec_sel(vb1, vb0, v_cmp01);

        const vector signed short v_diff89 = vec_sub(vb8, vb9);

        const vector unsigned short v_cmp89 =

            (const vector unsigned short) vec_cmplt(vec_abs(v_diff89), vqp);

        const vector signed short v_last = vec_sel(vb8, vb9, v_cmp89);



        const vector signed short temp01 = vec_mladd(v_first, (vector signed short)v_4, vb1);

        const vector signed short temp02 = vec_add(vb2, vb3);

        const vector signed short temp03 = vec_add(temp01, (vector signed short)v_4);

        const vector signed short v_sumsB0 = vec_add(temp02, temp03);



        const vector signed short temp11 = vec_sub(v_sumsB0, v_first);

        const vector signed short v_sumsB1 = vec_add(temp11, vb4);



        const vector signed short temp21 = vec_sub(v_sumsB1, v_first);

        const vector signed short v_sumsB2 = vec_add(temp21, vb5);



        const vector signed short temp31 = vec_sub(v_sumsB2, v_first);

        const vector signed short v_sumsB3 = vec_add(temp31, vb6);



        const vector signed short temp41 = vec_sub(v_sumsB3, v_first);

        const vector signed short v_sumsB4 = vec_add(temp41, vb7);



        const vector signed short temp51 = vec_sub(v_sumsB4, vb1);

        const vector signed short v_sumsB5 = vec_add(temp51, vb8);



        const vector signed short temp61 = vec_sub(v_sumsB5, vb2);

        const vector signed short v_sumsB6 = vec_add(temp61, v_last);



        const vector signed short temp71 = vec_sub(v_sumsB6, vb3);

        const vector signed short v_sumsB7 = vec_add(temp71, v_last);



        const vector signed short temp81 = vec_sub(v_sumsB7, vb4);

        const vector signed short v_sumsB8 = vec_add(temp81, v_last);



        const vector signed short temp91 = vec_sub(v_sumsB8, vb5);

        const vector signed short v_sumsB9 = vec_add(temp91, v_last);



    #define COMPUTE_VR(i, j, k)                                             \

        const vector signed short temps1##i =                               \

            vec_add(v_sumsB##i, v_sumsB##k);                                \

        const vector signed short temps2##i =                               \

            vec_mladd(vb##j, (vector signed short)v_2, temps1##i);          \

        const vector signed short  vr##j = vec_sra(temps2##i, v_4)



        COMPUTE_VR(0, 1, 2);

        COMPUTE_VR(1, 2, 3);

        COMPUTE_VR(2, 3, 4);

        COMPUTE_VR(3, 4, 5);

        COMPUTE_VR(4, 5, 6);

        COMPUTE_VR(5, 6, 7);

        COMPUTE_VR(6, 7, 8);

        COMPUTE_VR(7, 8, 9);



        const vector signed char neg1 = vec_splat_s8(-1);

        const vector unsigned char permHH = (const vector unsigned char){0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,

                                                                         0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};



#define PACK_AND_STORE(i)                                       \

{   const vector unsigned char perms##i =                       \

        vec_lvsr(i * stride, src2);                             \

    const vector unsigned char vf##i =                          \

        vec_packsu(vr##i, (vector signed short)zero);           \

    const vector unsigned char vg##i =                          \

        vec_perm(vf##i, vbT##i, permHH);                        \

    const vector unsigned char mask##i =                        \

        vec_perm((vector unsigned char)zero, (vector unsigned char)neg1, perms##i); \

    const vector unsigned char vg2##i =                         \

        vec_perm(vg##i, vg##i, perms##i);                       \

    const vector unsigned char svA##i =                         \

        vec_sel(vbA##i, vg2##i, mask##i);                       \

    const vector unsigned char svB##i =                         \

        vec_sel(vg2##i, vbB##i, mask##i);                       \

    vec_st(svA##i, i * stride, src2);                           \

    vec_st(svB##i, i * stride + 16, src2);}



#define PACK_AND_STORE_ALIGNED(i)                               \

{   const vector unsigned char vf##i =                          \

        vec_packsu(vr##i, (vector signed short)zero);           \

    const vector unsigned char vg##i =                          \

        vec_perm(vf##i, vbT##i, permHH);                        \

    vec_st(vg##i, i * stride, src2);}



        /* Special-casing the aligned case is worthwhile, as all calls from

         * the (transposed) horizontable deblocks will be aligned, in addition

         * to the naturally aligned vertical deblocks. */

        if (properStride && srcAlign) {

            PACK_AND_STORE_ALIGNED(1)

            PACK_AND_STORE_ALIGNED(2)

            PACK_AND_STORE_ALIGNED(3)

            PACK_AND_STORE_ALIGNED(4)

            PACK_AND_STORE_ALIGNED(5)

            PACK_AND_STORE_ALIGNED(6)

            PACK_AND_STORE_ALIGNED(7)

            PACK_AND_STORE_ALIGNED(8)

        } else {

            PACK_AND_STORE(1)

            PACK_AND_STORE(2)

            PACK_AND_STORE(3)

            PACK_AND_STORE(4)

            PACK_AND_STORE(5)

            PACK_AND_STORE(6)

            PACK_AND_STORE(7)

            PACK_AND_STORE(8)

        }

    #undef PACK_AND_STORE

    #undef PACK_AND_STORE_ALIGNED

    }

}
