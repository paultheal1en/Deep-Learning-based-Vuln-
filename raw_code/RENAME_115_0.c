static inline void RENAME(rgb32tobgr24)(const uint8_t *src, uint8_t *dst, long src_size)

{

    uint8_t *dest = dst;

    const uint8_t *s = src;

    const uint8_t *end;

#if COMPILE_TEMPLATE_MMX

    const uint8_t *mm_end;

#endif

    end = s + src_size;

#if COMPILE_TEMPLATE_MMX

    __asm__ volatile(PREFETCH"    %0"::"m"(*s):"memory");

    mm_end = end - 31;

    while (s < mm_end) {

        __asm__ volatile(

            PREFETCH"    32%1           \n\t"

            "movq          %1, %%mm0    \n\t"

            "movq         8%1, %%mm1    \n\t"

            "movq        16%1, %%mm4    \n\t"

            "movq        24%1, %%mm5    \n\t"

            "movq       %%mm0, %%mm2    \n\t"

            "movq       %%mm1, %%mm3    \n\t"

            "movq       %%mm4, %%mm6    \n\t"

            "movq       %%mm5, %%mm7    \n\t"

            STORE_BGR24_MMX

            :"=m"(*dest)

            :"m"(*s)

            :"memory");

        dest += 24;

        s += 32;

    }

    __asm__ volatile(SFENCE:::"memory");

    __asm__ volatile(EMMS:::"memory");

#endif

    while (s < end) {

#if HAVE_BIGENDIAN

        /* RGB32 (= A,B,G,R) -> RGB24 (= R,G,B) */

        s++;

        dest[2] = *s++;

        dest[1] = *s++;

        dest[0] = *s++;

        dest += 3;

#else

        *dest++ = *s++;

        *dest++ = *s++;

        *dest++ = *s++;

        s++;

#endif

    }

}
