void idct_add_altivec(uint8_t* dest, int stride, vector_s16_t* block)

{

POWERPC_TBL_DECLARE(altivec_idct_add_num, 1);

#ifdef ALTIVEC_USE_REFERENCE_C_CODE

POWERPC_TBL_START_COUNT(altivec_idct_add_num, 1);

    void simple_idct_add(uint8_t *dest, int line_size, int16_t *block);

    simple_idct_add(dest, stride, (int16_t*)block);

POWERPC_TBL_STOP_COUNT(altivec_idct_add_num, 1);

#else /* ALTIVEC_USE_REFERENCE_C_CODE */

    vector_u8_t tmp;

    vector_s16_t tmp2, tmp3;

    vector_u8_t perm0;

    vector_u8_t perm1;

    vector_u8_t p0, p1, p;



POWERPC_TBL_START_COUNT(altivec_idct_add_num, 1);



    IDCT



    p0 = vec_lvsl (0, dest);

    p1 = vec_lvsl (stride, dest);

    p = vec_splat_u8 (-1);

    perm0 = vec_mergeh (p, p0);

    perm1 = vec_mergeh (p, p1);



#define ADD(dest,src,perm)						\

    /* *(uint64_t *)&tmp = *(uint64_t *)dest; */			\

    tmp = vec_ld (0, dest);						\

    tmp2 = (vector_s16_t)vec_perm (tmp, (vector_u8_t)zero, perm);	\

    tmp3 = vec_adds (tmp2, src);					\

    tmp = vec_packsu (tmp3, tmp3);					\

    vec_ste ((vector_u32_t)tmp, 0, (unsigned int *)dest);		\

    vec_ste ((vector_u32_t)tmp, 4, (unsigned int *)dest);



    ADD (dest, vx0, perm0)	dest += stride;

    ADD (dest, vx1, perm1)	dest += stride;

    ADD (dest, vx2, perm0)	dest += stride;

    ADD (dest, vx3, perm1)	dest += stride;

    ADD (dest, vx4, perm0)	dest += stride;

    ADD (dest, vx5, perm1)	dest += stride;

    ADD (dest, vx6, perm0)	dest += stride;

    ADD (dest, vx7, perm1)



POWERPC_TBL_STOP_COUNT(altivec_idct_add_num, 1);

#endif /* ALTIVEC_USE_REFERENCE_C_CODE */

}
