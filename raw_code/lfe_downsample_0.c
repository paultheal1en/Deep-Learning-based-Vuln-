static void lfe_downsample(DCAEncContext *c, const int32_t *input)

{

    /* FIXME: make 128x LFE downsampling possible */

    const int lfech = lfe_index[c->channel_config];

    int i, j, lfes;

    int32_t hist[512];

    int32_t accum;

    int hist_start = 0;



    for (i = 0; i < 512; i++)

        hist[i] = c->history[i][c->channels - 1];



    for (lfes = 0; lfes < DCA_LFE_SAMPLES; lfes++) {

        /* Calculate the convolution */

        accum = 0;



        for (i = hist_start, j = 0; i < 512; i++, j++)

            accum += mul32(hist[i], lfe_fir_64i[j]);

        for (i = 0; i < hist_start; i++, j++)

            accum += mul32(hist[i], lfe_fir_64i[j]);



        c->downsampled_lfe[lfes] = accum;



        /* Copy in 64 new samples from input */

        for (i = 0; i < 64; i++)

            hist[i + hist_start] = input[(lfes * 64 + i) * c->channels + lfech];



        hist_start = (hist_start + 64) & 511;

    }

}
