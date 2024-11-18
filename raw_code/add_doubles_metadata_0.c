static int add_doubles_metadata(const uint8_t **buf, int count,

                                const char *name, const char *sep,

                                TiffContext *s)

{

    char *ap;

    int i;

    double *dp = av_malloc(count * sizeof(double));

    if (!dp)

        return AVERROR(ENOMEM);



    for (i = 0; i < count; i++)

        dp[i] = tget_double(buf, s->le);

    ap = doubles2str(dp, count, sep);

    av_freep(&dp);

    if (!ap)

        return AVERROR(ENOMEM);

    av_dict_set(&s->picture.metadata, name, ap, AV_DICT_DONT_STRDUP_VAL);

    return 0;

}
