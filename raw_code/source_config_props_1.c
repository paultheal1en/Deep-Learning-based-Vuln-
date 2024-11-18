static int source_config_props(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    Frei0rContext *s = ctx->priv;



    if (av_image_check_size(s->w, s->h, 0, ctx) < 0)

        return AVERROR(EINVAL);

    outlink->w = s->w;

    outlink->h = s->h;

    outlink->time_base = s->time_base;





    if (!(s->instance = s->construct(outlink->w, outlink->h))) {

        av_log(ctx, AV_LOG_ERROR, "Impossible to load frei0r instance");

        return AVERROR(EINVAL);

    }



    return set_params(ctx, s->params);

}