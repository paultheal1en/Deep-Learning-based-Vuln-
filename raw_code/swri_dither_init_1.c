int swri_dither_init(SwrContext *s, enum AVSampleFormat out_fmt, enum AVSampleFormat in_fmt)

{

    int i;

    double scale = 0;



    if (s->dither.method > SWR_DITHER_TRIANGULAR_HIGHPASS && s->dither.method <= SWR_DITHER_NS)

        return AVERROR(EINVAL);



    out_fmt = av_get_packed_sample_fmt(out_fmt);

    in_fmt  = av_get_packed_sample_fmt( in_fmt);



    if(in_fmt == AV_SAMPLE_FMT_FLT || in_fmt == AV_SAMPLE_FMT_DBL){

        if(out_fmt == AV_SAMPLE_FMT_S32) scale = 1.0/(1L<<31);

        if(out_fmt == AV_SAMPLE_FMT_S16) scale = 1.0/(1L<<15);

        if(out_fmt == AV_SAMPLE_FMT_U8 ) scale = 1.0/(1L<< 7);

    }

    if(in_fmt == AV_SAMPLE_FMT_S32 && out_fmt == AV_SAMPLE_FMT_S16) scale = 1L<<16;

    if(in_fmt == AV_SAMPLE_FMT_S32 && out_fmt == AV_SAMPLE_FMT_U8 ) scale = 1L<<24;

    if(in_fmt == AV_SAMPLE_FMT_S16 && out_fmt == AV_SAMPLE_FMT_U8 ) scale = 1L<<8;



    scale *= s->dither.scale;



    s->dither.ns_pos = 0;

    s->dither.noise_scale=   scale;

    s->dither.ns_scale   =   scale;

    s->dither.ns_scale_1 = 1/scale;

    memset(s->dither.ns_errors, 0, sizeof(s->dither.ns_errors));

    for (i=0; filters[i].coefs; i++) {

        const filter_t *f = &filters[i];

        if (fabs(s->out_sample_rate - f->rate) / f->rate <= .05 && f->name == s->dither.method) {

            int j;

            s->dither.ns_taps = f->len;

            for (j=0; j<f->len; j++)

                s->dither.ns_coeffs[j] = f->coefs[j];

            s->dither.ns_scale_1 *= 1 - exp(f->gain_cB * M_LN10 * 0.005) * 2 / (1<<(8*av_get_bytes_per_sample(out_fmt)));

            break;

        }

    }

    if (!filters[i].coefs && s->dither.method > SWR_DITHER_NS) {

        av_log(s, AV_LOG_WARNING, "Requested noise shaping dither not available at this sampling rate, using triangular hp dither\n");

        s->dither.method = SWR_DITHER_TRIANGULAR_HIGHPASS;

    }



    av_assert0(!s->preout.count);

    s->dither.noise = s->preout;

    s->dither.temp  = s->preout;

    if (s->dither.method > SWR_DITHER_NS) {

        s->dither.noise.bps = 4;

        s->dither.noise.fmt = AV_SAMPLE_FMT_FLTP;

        s->dither.noise_scale = 1;

    }



    return 0;

}
