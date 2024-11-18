int ff_fft_init(FFTContext *s, int nbits, int inverse)

{

    int i, j, m, n;

    float alpha, c1, s1, s2;

    int shuffle = 0;

    int av_unused has_vectors;



    s->nbits = nbits;

    n = 1 << nbits;



    s->exptab = av_malloc((n / 2) * sizeof(FFTComplex));

    if (!s->exptab)

        goto fail;

    s->revtab = av_malloc(n * sizeof(uint16_t));

    if (!s->revtab)

        goto fail;

    s->inverse = inverse;



    s2 = inverse ? 1.0 : -1.0;



    for(i=0;i<(n/2);i++) {

        alpha = 2 * M_PI * (float)i / (float)n;

        c1 = cos(alpha);

        s1 = sin(alpha) * s2;

        s->exptab[i].re = c1;

        s->exptab[i].im = s1;

    }

    s->fft_calc = ff_fft_calc_c;

    s->imdct_calc = ff_imdct_calc;

    s->imdct_half = ff_imdct_half;

    s->exptab1 = NULL;



#ifdef HAVE_MMX

    has_vectors = mm_support();

    shuffle = 1;

    if (has_vectors & MM_3DNOWEXT) {

        /* 3DNowEx for K7/K8 */

        s->imdct_calc = ff_imdct_calc_3dn2;


        s->fft_calc = ff_fft_calc_3dn2;

    } else if (has_vectors & MM_3DNOW) {

        /* 3DNow! for K6-2/3 */

        s->fft_calc = ff_fft_calc_3dn;

    } else if (has_vectors & MM_SSE) {

        /* SSE for P3/P4 */

        s->imdct_calc = ff_imdct_calc_sse;

        s->imdct_half = ff_imdct_half_sse;

        s->fft_calc = ff_fft_calc_sse;

    } else {

        shuffle = 0;

    }

#elif defined HAVE_ALTIVEC && !defined ALTIVEC_USE_REFERENCE_C_CODE

    has_vectors = mm_support();

    if (has_vectors & MM_ALTIVEC) {

        s->fft_calc = ff_fft_calc_altivec;

        shuffle = 1;

    }

#endif



    /* compute constant table for HAVE_SSE version */

    if (shuffle) {

        int np, nblocks, np2, l;

        FFTComplex *q;



        np = 1 << nbits;

        nblocks = np >> 3;

        np2 = np >> 1;

        s->exptab1 = av_malloc(np * 2 * sizeof(FFTComplex));

        if (!s->exptab1)

            goto fail;

        q = s->exptab1;

        do {

            for(l = 0; l < np2; l += 2 * nblocks) {

                *q++ = s->exptab[l];

                *q++ = s->exptab[l + nblocks];



                q->re = -s->exptab[l].im;

                q->im = s->exptab[l].re;

                q++;

                q->re = -s->exptab[l + nblocks].im;

                q->im = s->exptab[l + nblocks].re;

                q++;

            }

            nblocks = nblocks >> 1;

        } while (nblocks != 0);

        av_freep(&s->exptab);

    }



    /* compute bit reverse table */



    for(i=0;i<n;i++) {

        m=0;

        for(j=0;j<nbits;j++) {

            m |= ((i >> j) & 1) << (nbits-j-1);

        }

        s->revtab[i]=m;

    }

    return 0;

 fail:

    av_freep(&s->revtab);

    av_freep(&s->exptab);

    av_freep(&s->exptab1);

    return -1;

}