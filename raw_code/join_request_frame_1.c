static int join_request_frame(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    JoinContext *s       = ctx->priv;

    AVFilterBufferRef *buf;

    JoinBufferPriv *priv;

    int linesize   = INT_MAX;

    int perms      = ~0;

    int nb_samples;

    int i, j, ret;



    /* get a frame on each input */

    for (i = 0; i < ctx->nb_inputs; i++) {

        AVFilterLink *inlink = ctx->inputs[i];



        if (!s->input_frames[i] &&

            (ret = ff_request_frame(inlink)) < 0)

            return ret;



        /* request the same number of samples on all inputs */

        if (i == 0) {

            nb_samples = s->input_frames[0]->audio->nb_samples;



            for (j = 1; !i && j < ctx->nb_inputs; j++)

                ctx->inputs[j]->request_samples = nb_samples;

        }

    }



    for (i = 0; i < s->nb_channels; i++) {

        ChannelMap *ch = &s->channels[i];

        AVFilterBufferRef *cur_buf = s->input_frames[ch->input];



        s->data[i] = cur_buf->extended_data[ch->in_channel_idx];

        linesize   = FFMIN(linesize, cur_buf->linesize[0]);

        perms     &= cur_buf->perms;

    }



    buf = avfilter_get_audio_buffer_ref_from_arrays(s->data, linesize, perms,

                                                    nb_samples, outlink->format,

                                                    outlink->channel_layout);

    if (!buf)

        return AVERROR(ENOMEM);



    buf->buf->free = join_free_buffer;

    buf->pts       = s->input_frames[0]->pts;



    if (!(priv = av_mallocz(sizeof(*priv))))

        goto fail;

    if (!(priv->in_buffers = av_mallocz(sizeof(*priv->in_buffers) * ctx->nb_inputs)))

        goto fail;



    for (i = 0; i < ctx->nb_inputs; i++)

        priv->in_buffers[i] = s->input_frames[i];

    priv->nb_in_buffers = ctx->nb_inputs;

    buf->buf->priv      = priv;



    ff_filter_samples(outlink, buf);



    memset(s->input_frames, 0, sizeof(*s->input_frames) * ctx->nb_inputs);



    return 0;



fail:

    avfilter_unref_buffer(buf);

    if (priv)

        av_freep(&priv->in_buffers);

    av_freep(&priv);

    return AVERROR(ENOMEM);

}
