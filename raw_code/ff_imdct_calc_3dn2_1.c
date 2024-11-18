void ff_imdct_calc_3dn2(MDCTContext *s, FFTSample *output,

                        const FFTSample *input, FFTSample *tmp)

{

    long n8, n4, n2, n;

    x86_reg k;

    const uint16_t *revtab = s->fft.revtab;

    const FFTSample *tcos = s->tcos;

    const FFTSample *tsin = s->tsin;

    const FFTSample *in1, *in2;

    FFTComplex *z = (FFTComplex *)tmp;



    n = 1 << s->nbits;

    n2 = n >> 1;

    n4 = n >> 2;

    n8 = n >> 3;



    /* pre rotation */

    in1 = input;

    in2 = input + n2 - 1;

    for(k = 0; k < n4; k++) {

        // FIXME a single block is faster, but gcc 2.95 and 3.4.x on 32bit can't compile it

        asm volatile(

            "movd       %0, %%mm0 \n\t"

            "movd       %2, %%mm1 \n\t"

            "punpckldq  %1, %%mm0 \n\t"

            "punpckldq  %3, %%mm1 \n\t"

            "movq    %%mm0, %%mm2 \n\t"

            "pfmul   %%mm1, %%mm0 \n\t"

            "pswapd  %%mm1, %%mm1 \n\t"

            "pfmul   %%mm1, %%mm2 \n\t"

            "pfpnacc %%mm2, %%mm0 \n\t"

            ::"m"(in2[-2*k]), "m"(in1[2*k]),

              "m"(tcos[k]), "m"(tsin[k])

        );

        asm volatile(

            "movq    %%mm0, %0    \n\t"

            :"=m"(z[revtab[k]])

        );

    }



    ff_fft_calc(&s->fft, z);



    /* post rotation + reordering */

    for(k = 0; k < n4; k++) {

        asm volatile(

            "movq       %0, %%mm0 \n\t"

            "movd       %1, %%mm1 \n\t"

            "punpckldq  %2, %%mm1 \n\t"

            "movq    %%mm0, %%mm2 \n\t"

            "pfmul   %%mm1, %%mm0 \n\t"

            "pswapd  %%mm1, %%mm1 \n\t"

            "pfmul   %%mm1, %%mm2 \n\t"

            "pfpnacc %%mm2, %%mm0 \n\t"

            "movq    %%mm0, %0    \n\t"

            :"+m"(z[k])

            :"m"(tcos[k]), "m"(tsin[k])

        );

    }



    k = n-8;

    asm volatile("movd %0, %%mm7" ::"r"(1<<31));

    asm volatile(

        "1: \n\t"

        "movq    (%4,%0), %%mm0 \n\t" // z[n8+k]

        "neg %0 \n\t"

        "pswapd -8(%4,%0), %%mm1 \n\t" // z[n8-1-k]

        "movq      %%mm0, %%mm2 \n\t"

        "pxor      %%mm7, %%mm2 \n\t"

        "punpckldq %%mm1, %%mm2 \n\t"

        "pswapd    %%mm2, %%mm3 \n\t"

        "punpckhdq %%mm1, %%mm0 \n\t"

        "pswapd    %%mm0, %%mm4 \n\t"

        "pxor      %%mm7, %%mm0 \n\t"

        "pxor      %%mm7, %%mm4 \n\t"

        "movq      %%mm3, -8(%3,%0) \n\t" // output[n-2-2*k] = { z[n8-1-k].im, -z[n8+k].re }

        "movq      %%mm4, -8(%2,%0) \n\t" // output[n2-2-2*k]= { -z[n8-1-k].re, z[n8+k].im }

        "neg %0 \n\t"

        "movq      %%mm0, (%1,%0) \n\t"   // output[2*k]     = { -z[n8+k].im, z[n8-1-k].re }

        "movq      %%mm2, (%2,%0) \n\t"   // output[n2+2*k]  = { -z[n8+k].re, z[n8-1-k].im }

        "sub $8, %0 \n\t"

        "jge 1b \n\t"

        :"+r"(k)

        :"r"(output), "r"(output+n2), "r"(output+n), "r"(z+n8)

        :"memory"

    );

    asm volatile("femms");

}
