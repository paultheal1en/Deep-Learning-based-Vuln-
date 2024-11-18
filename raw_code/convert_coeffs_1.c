static int convert_coeffs(AVFilterContext *ctx, AVFilterLink *inlink)

{

    struct HeadphoneContext *s = ctx->priv;

    const int ir_len = s->ir_len;

    int nb_irs = s->nb_irs;

    int nb_input_channels = ctx->inputs[0]->channels;

    float gain_lin = expf((s->gain - 3 * nb_input_channels) / 20 * M_LN10);

    FFTComplex *data_hrtf_l = NULL;

    FFTComplex *data_hrtf_r = NULL;

    FFTComplex *fft_in_l = NULL;

    FFTComplex *fft_in_r = NULL;

    float *data_ir_l = NULL;

    float *data_ir_r = NULL;

    int offset = 0;

    int n_fft;

    int i, j;



    s->buffer_length = 1 << (32 - ff_clz(s->ir_len));

    s->n_fft = n_fft = 1 << (32 - ff_clz(s->ir_len + inlink->sample_rate));



    if (s->type == FREQUENCY_DOMAIN) {

        fft_in_l = av_calloc(n_fft, sizeof(*fft_in_l));

        fft_in_r = av_calloc(n_fft, sizeof(*fft_in_r));

        if (!fft_in_l || !fft_in_r) {

            return AVERROR(ENOMEM);

        }



        av_fft_end(s->fft[0]);

        av_fft_end(s->fft[1]);

        s->fft[0] = av_fft_init(log2(s->n_fft), 0);

        s->fft[1] = av_fft_init(log2(s->n_fft), 0);

        av_fft_end(s->ifft[0]);

        av_fft_end(s->ifft[1]);

        s->ifft[0] = av_fft_init(log2(s->n_fft), 1);

        s->ifft[1] = av_fft_init(log2(s->n_fft), 1);



        if (!s->fft[0] || !s->fft[1] || !s->ifft[0] || !s->ifft[1]) {

            av_log(ctx, AV_LOG_ERROR, "Unable to create FFT contexts of size %d.\n", s->n_fft);

            return AVERROR(ENOMEM);

        }

    }



    s->data_ir[0] = av_calloc(FFALIGN(s->ir_len, 16), sizeof(float) * s->nb_irs);

    s->data_ir[1] = av_calloc(FFALIGN(s->ir_len, 16), sizeof(float) * s->nb_irs);

    s->delay[0] = av_malloc_array(s->nb_irs, sizeof(float));

    s->delay[1] = av_malloc_array(s->nb_irs, sizeof(float));



    if (s->type == TIME_DOMAIN) {

        s->ringbuffer[0] = av_calloc(s->buffer_length, sizeof(float) * nb_input_channels);

        s->ringbuffer[1] = av_calloc(s->buffer_length, sizeof(float) * nb_input_channels);

    } else {

        s->ringbuffer[0] = av_calloc(s->buffer_length, sizeof(float));

        s->ringbuffer[1] = av_calloc(s->buffer_length, sizeof(float));

        s->temp_fft[0] = av_malloc_array(s->n_fft, sizeof(FFTComplex));

        s->temp_fft[1] = av_malloc_array(s->n_fft, sizeof(FFTComplex));

        if (!s->temp_fft[0] || !s->temp_fft[1])

            return AVERROR(ENOMEM);

    }



    if (!s->data_ir[0] || !s->data_ir[1] ||

        !s->ringbuffer[0] || !s->ringbuffer[1])

        return AVERROR(ENOMEM);



    s->in[0].frame = ff_get_audio_buffer(ctx->inputs[0], s->size);

    if (!s->in[0].frame)

        return AVERROR(ENOMEM);

    for (i = 0; i < s->nb_irs; i++) {

        s->in[i + 1].frame = ff_get_audio_buffer(ctx->inputs[i + 1], s->ir_len);

        if (!s->in[i + 1].frame)

            return AVERROR(ENOMEM);

    }



    if (s->type == TIME_DOMAIN) {

        s->temp_src[0] = av_calloc(FFALIGN(ir_len, 16), sizeof(float));

        s->temp_src[1] = av_calloc(FFALIGN(ir_len, 16), sizeof(float));



        data_ir_l = av_calloc(nb_irs * FFALIGN(ir_len, 16), sizeof(*data_ir_l));

        data_ir_r = av_calloc(nb_irs * FFALIGN(ir_len, 16), sizeof(*data_ir_r));

        if (!data_ir_r || !data_ir_l || !s->temp_src[0] || !s->temp_src[1]) {

            av_free(data_ir_l);

            av_free(data_ir_r);

            return AVERROR(ENOMEM);

        }

    } else {

        data_hrtf_l = av_malloc_array(n_fft, sizeof(*data_hrtf_l) * nb_irs);

        data_hrtf_r = av_malloc_array(n_fft, sizeof(*data_hrtf_r) * nb_irs);

        if (!data_hrtf_r || !data_hrtf_l) {

            av_free(data_hrtf_l);

            av_free(data_hrtf_r);

            return AVERROR(ENOMEM);

        }

    }



    for (i = 0; i < s->nb_irs; i++) {

        int len = s->in[i + 1].ir_len;

        int delay_l = s->in[i + 1].delay_l;

        int delay_r = s->in[i + 1].delay_r;

        int idx = -1;

        float *ptr;



        for (j = 0; j < inlink->channels; j++) {

            if (s->mapping[i] < 0) {

                continue;

            }



            if ((av_channel_layout_extract_channel(inlink->channel_layout, j)) == (1LL << s->mapping[i])) {

                idx = j;

                break;

            }

        }

        if (idx == -1)

            continue;



        av_audio_fifo_read(s->in[i + 1].fifo, (void **)s->in[i + 1].frame->extended_data, len);

        ptr = (float *)s->in[i + 1].frame->extended_data[0];



        if (s->type == TIME_DOMAIN) {

            offset = idx * FFALIGN(len, 16);

            for (j = 0; j < len; j++) {

                data_ir_l[offset + j] = ptr[len * 2 - j * 2 - 2] * gain_lin;

                data_ir_r[offset + j] = ptr[len * 2 - j * 2 - 1] * gain_lin;

            }

        } else {

            memset(fft_in_l, 0, n_fft * sizeof(*fft_in_l));

            memset(fft_in_r, 0, n_fft * sizeof(*fft_in_r));



            offset = idx * n_fft;

            for (j = 0; j < len; j++) {

                fft_in_l[delay_l + j].re = ptr[j * 2    ] * gain_lin;

                fft_in_r[delay_r + j].re = ptr[j * 2 + 1] * gain_lin;

            }



            av_fft_permute(s->fft[0], fft_in_l);

            av_fft_calc(s->fft[0], fft_in_l);

            memcpy(data_hrtf_l + offset, fft_in_l, n_fft * sizeof(*fft_in_l));

            av_fft_permute(s->fft[0], fft_in_r);

            av_fft_calc(s->fft[0], fft_in_r);

            memcpy(data_hrtf_r + offset, fft_in_r, n_fft * sizeof(*fft_in_r));

        }

    }



    if (s->type == TIME_DOMAIN) {

        memcpy(s->data_ir[0], data_ir_l, sizeof(float) * nb_irs * FFALIGN(ir_len, 16));

        memcpy(s->data_ir[1], data_ir_r, sizeof(float) * nb_irs * FFALIGN(ir_len, 16));



        av_freep(&data_ir_l);

        av_freep(&data_ir_r);

    } else {

        s->data_hrtf[0] = av_malloc_array(n_fft * s->nb_irs, sizeof(FFTComplex));

        s->data_hrtf[1] = av_malloc_array(n_fft * s->nb_irs, sizeof(FFTComplex));

        if (!s->data_hrtf[0] || !s->data_hrtf[1]) {

            av_freep(&data_hrtf_l);

            av_freep(&data_hrtf_r);

            av_freep(&fft_in_l);

            av_freep(&fft_in_r);

            return AVERROR(ENOMEM);

        }



        memcpy(s->data_hrtf[0], data_hrtf_l,

            sizeof(FFTComplex) * nb_irs * n_fft);

        memcpy(s->data_hrtf[1], data_hrtf_r,

            sizeof(FFTComplex) * nb_irs * n_fft);



        av_freep(&data_hrtf_l);

        av_freep(&data_hrtf_r);



        av_freep(&fft_in_l);

        av_freep(&fft_in_r);

    }



    s->have_hrirs = 1;



    return 0;

}
