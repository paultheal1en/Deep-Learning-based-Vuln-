AVFilterFormats *ff_make_format_list(const int *fmts)

{

    AVFilterFormats *formats;

    int count;



    for (count = 0; fmts[count] != -1; count++)

        ;



    formats               = av_mallocz(sizeof(*formats));

    if (count)

        formats->formats  = av_malloc(sizeof(*formats->formats) * count);

    formats->nb_formats = count;

    memcpy(formats->formats, fmts, sizeof(*formats->formats) * count);



    return formats;

}
