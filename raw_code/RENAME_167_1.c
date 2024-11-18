static inline void RENAME(rgb15to16)(const uint8_t *src, uint8_t *dst, int src_size)

{

    register const uint8_t* s=src;

    register uint8_t* d=dst;

    register const uint8_t *end;

    const uint8_t *mm_end;

    end = s + src_size;

    __asm__ volatile(PREFETCH"    %0"::"m"(*s));

    __asm__ volatile("movq        %0, %%mm4"::"m"(mask15s));

    mm_end = end - 15;

    while (s<mm_end) {

        __asm__ volatile(

            PREFETCH"  32%1         \n\t"

            "movq        %1, %%mm0  \n\t"

            "movq       8%1, %%mm2  \n\t"

            "movq     %%mm0, %%mm1  \n\t"

            "movq     %%mm2, %%mm3  \n\t"

            "pand     %%mm4, %%mm0  \n\t"

            "pand     %%mm4, %%mm2  \n\t"

            "paddw    %%mm1, %%mm0  \n\t"

            "paddw    %%mm3, %%mm2  \n\t"

            MOVNTQ"   %%mm0,  %0    \n\t"

            MOVNTQ"   %%mm2, 8%0"

            :"=m"(*d)

            :"m"(*s)

        );

        d+=16;

        s+=16;

    }

    __asm__ volatile(SFENCE:::"memory");

    __asm__ volatile(EMMS:::"memory");

    mm_end = end - 3;

    while (s < mm_end) {

        register unsigned x= *((const uint32_t *)s);

        *((uint32_t *)d) = (x&0x7FFF7FFF) + (x&0x7FE07FE0);

        d+=4;

        s+=4;

    }

    if (s < end) {

        register unsigned short x= *((const uint16_t *)s);

        *((uint16_t *)d) = (x&0x7FFF) + (x&0x7FE0);

    }

}
