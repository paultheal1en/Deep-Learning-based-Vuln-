static int split_init(AVFilterContext *ctx, const char *args, void *opaque)

{

    int i, nb_outputs = 2;



    if (args) {

        nb_outputs = strtol(args, NULL, 0);

        if (nb_outputs <= 0) {

            av_log(ctx, AV_LOG_ERROR, "Invalid number of outputs specified: %d.\n",

                   nb_outputs);

            return AVERROR(EINVAL);

        }

    }



    for (i = 0; i < nb_outputs; i++) {

        char name[32];

        AVFilterPad pad = { 0 };



        snprintf(name, sizeof(name), "output%d", i);

        pad.type = !strcmp(ctx->name, "split") ? AVMEDIA_TYPE_VIDEO : AVMEDIA_TYPE_AUDIO;

        pad.name = av_strdup(name);



        avfilter_insert_outpad(ctx, i, &pad);

    }



    return 0;

}
