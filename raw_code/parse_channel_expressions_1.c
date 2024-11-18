static int parse_channel_expressions(AVFilterContext *ctx,

                                     int expected_nb_channels)

{

    EvalContext *eval = ctx->priv;

    char *args1 = av_strdup(eval->exprs);

    char *expr, *last_expr, *buf;

    double (* const *func1)(void *, double) = NULL;

    const char * const *func1_names = NULL;

    int i, ret = 0;



    if (!args1)

        return AVERROR(ENOMEM);



    if (!eval->exprs) {

        av_log(ctx, AV_LOG_ERROR, "Channels expressions list is empty\n");

        return AVERROR(EINVAL);

    }



    if (!strcmp(ctx->filter->name, "aeval")) {

        func1 = aeval_func1;

        func1_names = aeval_func1_names;

    }



#define ADD_EXPRESSION(expr_) do {                                      \

        if (!av_dynarray2_add((void **)&eval->expr, &eval->nb_channels, \

                              sizeof(*eval->expr), NULL)) {             \

            ret = AVERROR(ENOMEM);                                      \

            goto end;                                                   \

        }                                                               \

        eval->expr[eval->nb_channels-1] = NULL;                         \

        ret = av_expr_parse(&eval->expr[eval->nb_channels - 1], expr_,  \

                            var_names, func1_names, func1,              \

                            NULL, NULL, 0, ctx);                        \

        if (ret < 0)                                                    \

            goto end;                                                   \

    } while (0)



    /* reset expressions */

    for (i = 0; i < eval->nb_channels; i++) {

        av_expr_free(eval->expr[i]);

        eval->expr[i] = NULL;

    }

    av_freep(&eval->expr);

    eval->nb_channels = 0;



    buf = args1;

    while (expr = av_strtok(buf, "|", &buf)) {

        ADD_EXPRESSION(expr);

        last_expr = expr;

    }



    if (expected_nb_channels > eval->nb_channels)

        for (i = eval->nb_channels; i < expected_nb_channels; i++)

            ADD_EXPRESSION(last_expr);



    if (expected_nb_channels > 0 && eval->nb_channels != expected_nb_channels) {

        av_log(ctx, AV_LOG_ERROR,

               "Mismatch between the specified number of channel expressions '%d' "

               "and the number of expected output channels '%d' for the specified channel layout\n",

               eval->nb_channels, expected_nb_channels);

        ret = AVERROR(EINVAL);

        goto end;

    }



end:

    av_free(args1);

    return ret;

}
