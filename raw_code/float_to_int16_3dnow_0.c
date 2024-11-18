static void float_to_int16_3dnow(int16_t *dst, const float *src, int len){

    // not bit-exact: pf2id uses different rounding than C and SSE

    int i;

    for(i=0; i<len; i+=4) {

        asm volatile(

            "pf2id       %1, %%mm0 \n\t"

            "pf2id       %2, %%mm1 \n\t"

            "packssdw %%mm1, %%mm0 \n\t"

            "movq     %%mm0, %0    \n\t"

            :"=m"(dst[i])

            :"m"(src[i]), "m"(src[i+2])

        );

    }

    asm volatile("femms");

}
