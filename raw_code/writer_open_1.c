static int writer_open(WriterContext **wctx, const Writer *writer, const char *args,

                       const struct section *sections, int nb_sections)

{

    int i, ret = 0;



    if (!(*wctx = av_malloc(sizeof(WriterContext)))) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    if (!((*wctx)->priv = av_mallocz(writer->priv_size))) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    (*wctx)->class = &writer_class;

    (*wctx)->writer = writer;

    (*wctx)->level = -1;

    (*wctx)->sections = sections;

    (*wctx)->nb_sections = nb_sections;



    if (writer->priv_class) {

        void *priv_ctx = (*wctx)->priv;

        *((const AVClass **)priv_ctx) = writer->priv_class;

        av_opt_set_defaults(priv_ctx);



        if (args &&

            (ret = av_set_options_string(priv_ctx, args, "=", ":")) < 0)

            goto fail;

    }



    for (i = 0; i < SECTION_MAX_NB_LEVELS; i++)

        av_bprint_init(&(*wctx)->section_pbuf[i], 1, AV_BPRINT_SIZE_UNLIMITED);



    if ((*wctx)->writer->init)

        ret = (*wctx)->writer->init(*wctx);

    if (ret < 0)

        goto fail;



    return 0;



fail:

    writer_close(wctx);

    return ret;

}
