static inline void RENAME(duplicate)(uint8_t src[], int stride)

{

#if TEMPLATE_PP_MMX

    __asm__ volatile(

        "movq (%0), %%mm0               \n\t"

        "add %1, %0                     \n\t"

        "movq %%mm0, (%0)               \n\t"

        "movq %%mm0, (%0, %1)           \n\t"

        "movq %%mm0, (%0, %1, 2)        \n\t"

        : "+r" (src)

        : "r" ((x86_reg)-stride)

    );

#else

    int i;

    uint8_t *p=src;

    for(i=0; i<3; i++){

        p-= stride;

        memcpy(p, src, 8);

    }

#endif

}
