void avg_pixels16_altivec(uint8_t *block, const uint8_t *pixels, int line_size, int h)

{

POWERPC_TBL_DECLARE(altivec_avg_pixels16_num, 1);

#ifdef ALTIVEC_USE_REFERENCE_C_CODE

    int i;



POWERPC_TBL_START_COUNT(altivec_avg_pixels16_num, 1);



    for(i=0; i<h; i++) {

      op_avg(*((uint32_t*)(block)),(((const struct unaligned_32 *)(pixels))->l));

      op_avg(*((uint32_t*)(block+4)),(((const struct unaligned_32 *)(pixels+4))->l));

      op_avg(*((uint32_t*)(block+8)),(((const struct unaligned_32 *)(pixels+8))->l));

      op_avg(*((uint32_t*)(block+12)),(((const struct unaligned_32 *)(pixels+12))->l));

      pixels+=line_size;

      block +=line_size;

    }



POWERPC_TBL_STOP_COUNT(altivec_avg_pixels16_num, 1);



#else /* ALTIVEC_USE_REFERENCE_C_CODE */

    register vector unsigned char pixelsv1, pixelsv2, pixelsv, blockv;

    register vector unsigned char perm = vec_lvsl(0, pixels);

    int i;



POWERPC_TBL_START_COUNT(altivec_avg_pixels16_num, 1);



    for(i=0; i<h; i++) {

      pixelsv1 = vec_ld(0, (unsigned char*)pixels);

      pixelsv2 = vec_ld(16, (unsigned char*)pixels);

      blockv = vec_ld(0, block);

      pixelsv = vec_perm(pixelsv1, pixelsv2, perm);

      blockv = vec_avg(blockv,pixelsv);

      vec_st(blockv, 0, (unsigned char*)block);

      pixels+=line_size;

      block +=line_size;

    }



POWERPC_TBL_STOP_COUNT(altivec_avg_pixels16_num, 1);



#endif /* ALTIVEC_USE_REFERENCE_C_CODE */

}
