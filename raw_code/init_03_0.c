static av_cold int init(AVFilterContext *ctx)

{

    FormatContext *s = ctx->priv;

    char *cur, *sep;

    int nb_formats = 1;

    int i;

    int ret;



    /* count the formats */

    cur = s->pix_fmts;

    while ((cur = strchr(cur, '|'))) {

        nb_formats++;

        if (*cur)

            cur++;

    }



    s->formats = av_malloc_array(nb_formats + 1, sizeof(*s->formats));

    if (!s->formats)

        return AVERROR(ENOMEM);



    if (!s->pix_fmts)

        return AVERROR(EINVAL);



    /* parse the list of formats */

    cur = s->pix_fmts;

    for (i = 0; i < nb_formats; i++) {

        sep = strchr(cur, '|');

        if (sep)

            *sep++ = 0;



        if ((ret = ff_parse_pixel_format(&s->formats[i], cur, ctx)) < 0)

            return ret;



        cur = sep;

    }

    s->formats[nb_formats] = AV_PIX_FMT_NONE;



    if (!strcmp(ctx->filter->name, "noformat")) {

        const AVPixFmtDescriptor *desc = NULL;

        enum AVPixelFormat *formats_allowed;

        int nb_formats_lavu = 0, nb_formats_allowed = 0;



        /* count the formats known to lavu */

        while ((desc = av_pix_fmt_desc_next(desc)))

            nb_formats_lavu++;



        formats_allowed = av_malloc_array(nb_formats_lavu + 1, sizeof(*formats_allowed));

        if (!formats_allowed)

            return AVERROR(ENOMEM);



        /* for each format known to lavu, check if it's in the list of

         * forbidden formats */

        while ((desc = av_pix_fmt_desc_next(desc))) {

            enum AVPixelFormat pix_fmt = av_pix_fmt_desc_get_id(desc);



            for (i = 0; i < nb_formats; i++) {

                if (s->formats[i] == pix_fmt)

                    break;

            }

            if (i < nb_formats)

                continue;



            formats_allowed[nb_formats_allowed++] = pix_fmt;

        }

        formats_allowed[nb_formats_allowed] = AV_PIX_FMT_NONE;

        av_freep(&s->formats);

        s->formats = formats_allowed;

    }



    return 0;

}
