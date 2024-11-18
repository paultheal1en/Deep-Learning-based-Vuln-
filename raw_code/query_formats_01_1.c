static int query_formats(AVFilterContext *ctx)

{

    AVFilterFormats *formats = NULL;

    int fmt;



    for (fmt = 0; fmt < AV_PIX_FMT_NB; fmt++) {

        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(fmt);

        if (!(desc->flags & PIX_FMT_PAL ||

            fmt == AV_PIX_FMT_NV21 ||

            fmt == AV_PIX_FMT_NV12))

            ff_add_format(&formats, fmt);

    }



    ff_set_common_formats(ctx, formats);

    return 0;

}
