static int parse_psfile(AVFilterContext *ctx, const char *fname)

{

    CurvesContext *curves = ctx->priv;

    uint8_t *buf;

    size_t size;

    int i, ret, av_unused(version), nb_curves;

    AVBPrint ptstr;

    static const int comp_ids[] = {3, 0, 1, 2};



    av_bprint_init(&ptstr, 0, AV_BPRINT_SIZE_AUTOMATIC);



    ret = av_file_map(fname, &buf, &size, 0, NULL);

    if (ret < 0)

        return ret;



#define READ16(dst) do {                \

    if (size < 2)                       \

        return AVERROR_INVALIDDATA;     \

    dst = AV_RB16(buf);                 \

    buf  += 2;                          \

    size -= 2;                          \

} while (0)



    READ16(version);

    READ16(nb_curves);

    for (i = 0; i < FFMIN(nb_curves, FF_ARRAY_ELEMS(comp_ids)); i++) {

        int nb_points, n;

        av_bprint_clear(&ptstr);

        READ16(nb_points);

        for (n = 0; n < nb_points; n++) {

            int y, x;

            READ16(y);

            READ16(x);

            av_bprintf(&ptstr, "%f/%f ", x / 255., y / 255.);

        }

        if (*ptstr.str) {

            char **pts = &curves->comp_points_str[comp_ids[i]];

            if (!*pts) {

                *pts = av_strdup(ptstr.str);

                av_log(ctx, AV_LOG_DEBUG, "curves %d (intid=%d) [%d points]: [%s]\n",

                       i, comp_ids[i], nb_points, *pts);

                if (!*pts) {

                    ret = AVERROR(ENOMEM);

                    goto end;

                }

            }

        }

    }

end:

    av_bprint_finalize(&ptstr, NULL);

    av_file_unmap(buf, size);

    return ret;

}
