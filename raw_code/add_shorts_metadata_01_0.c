static int add_shorts_metadata(const uint8_t **buf, int count, const char *name,

                               const char *sep, TiffContext *s)

{

    char *ap;

    int i;

    int *sp = av_malloc(count * sizeof(int));

    if (!sp)

        return AVERROR(ENOMEM);



    for (i = 0; i < count; i++)

        sp[i] = tget_short(buf, s->le);

    ap = shorts2str(sp, count, sep);

    av_freep(&sp);

    if (!ap)

        return AVERROR(ENOMEM);

    av_dict_set(&s->picture.metadata, name, ap, AV_DICT_DONT_STRDUP_VAL);

    return 0;

}
