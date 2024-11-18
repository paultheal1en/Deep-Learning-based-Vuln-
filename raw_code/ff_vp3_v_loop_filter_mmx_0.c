void ff_vp3_v_loop_filter_mmx(uint8_t *src, int stride, int *bounding_values)

{

    __asm__ volatile(

        "movq          %0, %%mm6 \n\t"

        "movq          %1, %%mm4 \n\t"

        "movq          %2, %%mm2 \n\t"

        "movq          %3, %%mm1 \n\t"



        VP3_LOOP_FILTER(%4)



        "movq       %%mm4, %1    \n\t"

        "movq       %%mm3, %2    \n\t"



        : "+m" (*(uint64_t*)(src - 2*stride)),

          "+m" (*(uint64_t*)(src - 1*stride)),

          "+m" (*(uint64_t*)(src + 0*stride)),

          "+m" (*(uint64_t*)(src + 1*stride))

        : "m"(*(uint64_t*)(bounding_values+129))

    );

}
