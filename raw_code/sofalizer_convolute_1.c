static int sofalizer_convolute(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)

{

    SOFAlizerContext *s = ctx->priv;

    ThreadData *td = arg;

    AVFrame *in = td->in, *out = td->out;

    int offset = jobnr;

    int *write = &td->write[jobnr];

    const int *const delay = td->delay[jobnr];

    const float *const ir = td->ir[jobnr];

    int *n_clippings = &td->n_clippings[jobnr];

    float *ringbuffer = td->ringbuffer[jobnr];

    float *temp_src = td->temp_src[jobnr];

    const int n_samples = s->sofa.n_samples; /* length of one IR */

    const float *src = (const float *)in->data[0]; /* get pointer to audio input buffer */

    float *dst = (float *)out->data[0]; /* get pointer to audio output buffer */

    const int in_channels = s->n_conv; /* number of input channels */

    /* ring buffer length is: longest IR plus max. delay -> next power of 2 */

    const int buffer_length = s->buffer_length;

    /* -1 for AND instead of MODULO (applied to powers of 2): */

    const uint32_t modulo = (uint32_t)buffer_length - 1;

    float *buffer[16]; /* holds ringbuffer for each input channel */

    int wr = *write;

    int read;

    int i, l;



    dst += offset;

    for (l = 0; l < in_channels; l++) {

        /* get starting address of ringbuffer for each input channel */

        buffer[l] = ringbuffer + l * buffer_length;

    }



    for (i = 0; i < in->nb_samples; i++) {

        const float *temp_ir = ir; /* using same set of IRs for each sample */



        *dst = 0;

        for (l = 0; l < in_channels; l++) {

            /* write current input sample to ringbuffer (for each channel) */

            *(buffer[l] + wr) = src[l];

        }



        /* loop goes through all channels to be convolved */

        for (l = 0; l < in_channels; l++) {

            const float *const bptr = buffer[l];



            if (l == s->lfe_channel) {

                /* LFE is an input channel but requires no convolution */

                /* apply gain to LFE signal and add to output buffer */

                *dst += *(buffer[s->lfe_channel] + wr) * s->gain_lfe;

                temp_ir += n_samples;

                continue;

            }



            /* current read position in ringbuffer: input sample write position

             * - delay for l-th ch. + diff. betw. IR length and buffer length

             * (mod buffer length) */

            read = (wr - *(delay + l) - (n_samples - 1) + buffer_length) & modulo;



            if (read + n_samples < buffer_length) {

                memcpy(temp_src, bptr + read, n_samples * sizeof(*temp_src));

            } else {

                int len = FFMIN(n_samples - (read % n_samples), buffer_length - read);



                memcpy(temp_src, bptr + read, len * sizeof(*temp_src));

                memcpy(temp_src + len, bptr, (n_samples - len) * sizeof(*temp_src));

            }



            /* multiply signal and IR, and add up the results */

            dst[0] += s->fdsp->scalarproduct_float(temp_ir, temp_src, n_samples);

            temp_ir += n_samples;

        }



        /* clippings counter */

        if (fabs(*dst) > 1)

            *n_clippings += 1;



        /* move output buffer pointer by +2 to get to next sample of processed channel: */

        dst += 2;

        src += in_channels;

        wr   = (wr + 1) & modulo; /* update ringbuffer write position */

    }



    *write = wr; /* remember write position in ringbuffer for next call */



    return 0;

}
