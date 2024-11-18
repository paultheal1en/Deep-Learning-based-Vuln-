static int channelmap_config_input(AVFilterLink *inlink)

{

    AVFilterContext *ctx = inlink->dst;

    ChannelMapContext *s = ctx->priv;

    int i, err = 0;

    const char *channel_name;

    char layout_name[256];



    if (s->mode == MAP_PAIR_STR_INT || s->mode == MAP_PAIR_STR_STR) {

        for (i = 0; i < s->nch; i++) {

            s->map[i].in_channel_idx = av_get_channel_layout_channel_index(

                inlink->channel_layout, s->map[i].in_channel);

            if (s->map[i].in_channel_idx < 0) {

                channel_name = av_get_channel_name(s->map[i].in_channel);

                av_get_channel_layout_string(layout_name, sizeof(layout_name),

                                             0, inlink->channel_layout);

                av_log(ctx, AV_LOG_ERROR,

                       "input channel '%s' not available from input layout '%s'\n",

                       channel_name, layout_name);

                err = AVERROR(EINVAL);

            }

        }

    }



    return err;

}
