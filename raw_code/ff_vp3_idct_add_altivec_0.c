void ff_vp3_idct_add_altivec(uint8_t *dst, int stride, DCTELEM block[64])

{

    LOAD_ZERO;

    vec_u8 t, vdst;

    vec_s16 vdst_16;

    vec_u8 vdst_mask = vec_mergeh(vec_splat_u8(-1), vec_lvsl(0, dst));



    IDCT_START



    IDCT_1D(NOP, NOP)

    TRANSPOSE8(b0, b1, b2, b3, b4, b5, b6, b7);

    IDCT_1D(ADD8, SHIFT4)



#define ADD(a)\

    vdst = vec_ld(0, dst);\

    vdst_16 = (vec_s16)vec_perm(vdst, zero_u8v, vdst_mask);\

    vdst_16 = vec_adds(a, vdst_16);\

    t = vec_packsu(vdst_16, vdst_16);\

    vec_ste((vec_u32)t, 0, (unsigned int *)dst);\

    vec_ste((vec_u32)t, 4, (unsigned int *)dst);



    ADD(b0)     dst += stride;

    ADD(b1)     dst += stride;

    ADD(b2)     dst += stride;

    ADD(b3)     dst += stride;

    ADD(b4)     dst += stride;

    ADD(b5)     dst += stride;

    ADD(b6)     dst += stride;

    ADD(b7)

}
