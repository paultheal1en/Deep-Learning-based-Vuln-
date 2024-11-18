av_cold struct FFIIRFilterCoeffs* ff_iir_filter_init_coeffs(enum IIRFilterType filt_type,

                                                    enum IIRFilterMode filt_mode,

                                                    int order, float cutoff_ratio,

                                                    float stopband, float ripple)

{

    int i, j;

    FFIIRFilterCoeffs *c;

    double wa;

    double p[MAXORDER + 1][2];



    if(filt_type != FF_FILTER_TYPE_BUTTERWORTH || filt_mode != FF_FILTER_MODE_LOWPASS)

        return NULL;

    if(order <= 1 || (order & 1) || order > MAXORDER || cutoff_ratio >= 1.0)

        return NULL;



    c = av_malloc(sizeof(FFIIRFilterCoeffs));

    c->cx = av_malloc(sizeof(c->cx[0]) * ((order >> 1) + 1));

    c->cy = av_malloc(sizeof(c->cy[0]) * order);

    c->order = order;



    wa = 2 * tan(M_PI * 0.5 * cutoff_ratio);



    c->cx[0] = 1;

    for(i = 1; i < (order >> 1) + 1; i++)

        c->cx[i] = c->cx[i - 1] * (order - i + 1LL) / i;



    p[0][0] = 1.0;

    p[0][1] = 0.0;

    for(i = 1; i <= order; i++)

        p[i][0] = p[i][1] = 0.0;

    for(i = 0; i < order; i++){

        double zp[2];

        double th = (i + (order >> 1) + 0.5) * M_PI / order;

        double a_re, a_im, c_re, c_im;

        zp[0] = cos(th) * wa;

        zp[1] = sin(th) * wa;

        a_re = zp[0] + 2.0;

        c_re = zp[0] - 2.0;

        a_im =

        c_im = zp[1];

        zp[0] = (a_re * c_re + a_im * c_im) / (c_re * c_re + c_im * c_im);

        zp[1] = (a_im * c_re - a_re * c_im) / (c_re * c_re + c_im * c_im);



        for(j = order; j >= 1; j--)

        {

            a_re = p[j][0];

            a_im = p[j][1];

            p[j][0] = a_re*zp[0] - a_im*zp[1] + p[j-1][0];

            p[j][1] = a_re*zp[1] + a_im*zp[0] + p[j-1][1];

        }

        a_re    = p[0][0]*zp[0] - p[0][1]*zp[1];

        p[0][1] = p[0][0]*zp[1] + p[0][1]*zp[0];

        p[0][0] = a_re;

    }

    c->gain = p[order][0];

    for(i = 0; i < order; i++){

        c->gain += p[i][0];

        c->cy[i] = (-p[i][0] * p[order][0] + -p[i][1] * p[order][1]) /

                   (p[order][0] * p[order][0] + p[order][1] * p[order][1]);

    }

    c->gain /= 1 << order;



    return c;

}
