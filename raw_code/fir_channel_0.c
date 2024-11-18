static int fir_channel(AVFilterContext *ctx, void *arg, int ch, int nb_jobs)

{

    AudioFIRContext *s = ctx->priv;

    const float *src = (const float *)s->in[0]->extended_data[ch];

    int index1 = (s->index + 1) % 3;

    int index2 = (s->index + 2) % 3;

    float *sum = s->sum[ch];

    AVFrame *out = arg;

    float *block;

    float *dst;

    int n, i, j;



    memset(sum, 0, sizeof(*sum) * s->fft_length);

    block = s->block[ch] + s->part_index * s->block_size;

    memset(block, 0, sizeof(*block) * s->fft_length);



    s->fdsp->vector_fmul_scalar(block + s->part_size, src, s->dry_gain, s->nb_samples);

    emms_c();



    av_rdft_calc(s->rdft[ch], block);

    block[2 * s->part_size] = block[1];

    block[1] = 0;



    j = s->part_index;



    for (i = 0; i < s->nb_partitions; i++) {

        const int coffset = i * s->coeff_size;

        const FFTComplex *coeff = s->coeff[ch * !s->one2many] + coffset;



        block = s->block[ch] + j * s->block_size;

        s->fcmul_add(sum, block, (const float *)coeff, s->part_size);



        if (j == 0)

            j = s->nb_partitions;

        j--;

    }



    sum[1] = sum[2 * s->part_size];

    av_rdft_calc(s->irdft[ch], sum);



    dst = (float *)s->buffer->extended_data[ch] + index1 * s->part_size;

    for (n = 0; n < s->part_size; n++) {

        dst[n] += sum[n];

    }



    dst = (float *)s->buffer->extended_data[ch] + index2 * s->part_size;



    memcpy(dst, sum + s->part_size, s->part_size * sizeof(*dst));



    dst = (float *)s->buffer->extended_data[ch] + s->index * s->part_size;



    if (out) {

        float *ptr = (float *)out->extended_data[ch];

        s->fdsp->vector_fmul_scalar(ptr, dst, s->gain * s->wet_gain, out->nb_samples);

        emms_c();

    }



    return 0;

}
