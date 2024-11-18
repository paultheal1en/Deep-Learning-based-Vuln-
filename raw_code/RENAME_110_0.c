static void RENAME(extract_even)(const uint8_t *src, uint8_t *dst, x86_reg count)

{

    dst +=   count;

    src += 2*count;

    count= - count;



#if COMPILE_TEMPLATE_MMX

    if(count <= -16) {

        count += 15;

        __asm__ volatile(

            "pcmpeqw       %%mm7, %%mm7        \n\t"

            "psrlw            $8, %%mm7        \n\t"

            "1:                                \n\t"

            "movq -30(%1, %0, 2), %%mm0        \n\t"

            "movq -22(%1, %0, 2), %%mm1        \n\t"

            "movq -14(%1, %0, 2), %%mm2        \n\t"

            "movq  -6(%1, %0, 2), %%mm3        \n\t"

            "pand          %%mm7, %%mm0        \n\t"

            "pand          %%mm7, %%mm1        \n\t"

            "pand          %%mm7, %%mm2        \n\t"

            "pand          %%mm7, %%mm3        \n\t"

            "packuswb      %%mm1, %%mm0        \n\t"

            "packuswb      %%mm3, %%mm2        \n\t"

            MOVNTQ"        %%mm0,-15(%2, %0)   \n\t"

            MOVNTQ"        %%mm2,- 7(%2, %0)   \n\t"

            "add             $16, %0           \n\t"

            " js 1b                            \n\t"

            : "+r"(count)

            : "r"(src), "r"(dst)

        );

        count -= 15;

    }

#endif

    while(count<0) {

        dst[count]= src[2*count];

        count++;

    }

}
