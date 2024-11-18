int avfilter_default_query_formats(AVFilterContext *ctx)

{

    enum AVMediaType type = ctx->inputs [0] ? ctx->inputs [0]->type :

                            ctx->outputs[0] ? ctx->outputs[0]->type :

                            AVMEDIA_TYPE_VIDEO;



    avfilter_set_common_formats(ctx, avfilter_all_formats(type));

    return 0;

}
