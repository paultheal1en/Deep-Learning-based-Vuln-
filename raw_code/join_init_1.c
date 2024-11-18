static av_cold int join_init(AVFilterContext *ctx)

{

    JoinContext *s = ctx->priv;

    int ret, i;



    if (!(s->channel_layout = av_get_channel_layout(s->channel_layout_str))) {

        av_log(ctx, AV_LOG_ERROR, "Error parsing channel layout '%s'.\n",

               s->channel_layout_str);

        return AVERROR(EINVAL);

    }



    s->nb_channels  = av_get_channel_layout_nb_channels(s->channel_layout);

    s->channels     = av_mallocz_array(s->nb_channels, sizeof(*s->channels));

    s->buffers      = av_mallocz_array(s->nb_channels, sizeof(*s->buffers));

    s->input_frames = av_mallocz_array(s->inputs, sizeof(*s->input_frames));

    if (!s->channels || !s->buffers|| !s->input_frames)

        return AVERROR(ENOMEM);



    for (i = 0; i < s->nb_channels; i++) {

        s->channels[i].out_channel = av_channel_layout_extract_channel(s->channel_layout, i);

        s->channels[i].input       = -1;

    }



    if ((ret = parse_maps(ctx)) < 0)

        return ret;



    for (i = 0; i < s->inputs; i++) {

        char name[32];

        AVFilterPad pad = { 0 };



        snprintf(name, sizeof(name), "input%d", i);

        pad.type           = AVMEDIA_TYPE_AUDIO;

        pad.name           = av_strdup(name);

        if (!pad.name)

            return AVERROR(ENOMEM);

        pad.filter_frame   = filter_frame;



        pad.needs_fifo = 1;



        ff_insert_inpad(ctx, i, &pad);

    }



    return 0;

}
