static int read_channels(AVFilterContext *ctx, int channels, uint8_t *item_str, int *nb, double **c, double **cache)

{

    char *p, *arg, *old_str, *prev_arg = NULL, *saveptr = NULL;

    int i, ret;



    p = old_str = av_strdup(item_str);

    if (!p)

        return AVERROR(ENOMEM);

    for (i = 0; i < channels; i++) {

        if (!(arg = av_strtok(p, "|", &saveptr)))

            arg = prev_arg;



        p = NULL;

        count_coefficients(arg, &nb[i]);

        cache[i] = av_calloc(nb[i], sizeof(cache[i]));

        c[i] = av_calloc(nb[i], sizeof(c[i]));

        if (!c[i] || !cache[i])

            return AVERROR(ENOMEM);



        ret = read_coefficients(ctx, arg, nb[i], c[i]);

        if (ret < 0)

            return ret;

        prev_arg = arg;

    }



    av_freep(&old_str);



    return 0;

}
