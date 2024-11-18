static void rdft_calc_c(RDFTContext *s, FFTSample *data)

{

    int i, i1, i2;

    FFTComplex ev, od;

    const int n = 1 << s->nbits;

    const float k1 = 0.5;

    const float k2 = 0.5 - s->inverse;

    const FFTSample *tcos = s->tcos;

    const FFTSample *tsin = s->tsin;



    if (!s->inverse) {

        s->fft.fft_permute(&s->fft, (FFTComplex*)data);

        s->fft.fft_calc(&s->fft, (FFTComplex*)data);

    }

    /* i=0 is a special case because of packing, the DC term is real, so we

       are going to throw the N/2 term (also real) in with it. */

    ev.re = data[0];

    data[0] = ev.re+data[1];

    data[1] = ev.re-data[1];

    for (i = 1; i < (n>>2); i++) {

        i1 = 2*i;

        i2 = n-i1;

        /* Separate even and odd FFTs */

        ev.re =  k1*(data[i1  ]+data[i2  ]);

        od.im = -k2*(data[i1  ]-data[i2  ]);

        ev.im =  k1*(data[i1+1]-data[i2+1]);

        od.re =  k2*(data[i1+1]+data[i2+1]);

        /* Apply twiddle factors to the odd FFT and add to the even FFT */

        data[i1  ] =  ev.re + od.re*tcos[i] - od.im*tsin[i];

        data[i1+1] =  ev.im + od.im*tcos[i] + od.re*tsin[i];

        data[i2  ] =  ev.re - od.re*tcos[i] + od.im*tsin[i];

        data[i2+1] = -ev.im + od.im*tcos[i] + od.re*tsin[i];

    }

    data[2*i+1]=s->sign_convention*data[2*i+1];

    if (s->inverse) {

        data[0] *= k1;

        data[1] *= k1;

        s->fft.fft_permute(&s->fft, (FFTComplex*)data);

        s->fft.fft_calc(&s->fft, (FFTComplex*)data);

    }

}
