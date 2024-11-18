static int channelmap_filter_samples(AVFilterLink *inlink, AVFilterBufferRef *buf)

{

    AVFilterContext  *ctx = inlink->dst;

    AVFilterLink *outlink = ctx->outputs[0];

    const ChannelMapContext *s = ctx->priv;

    const int nch_in = av_get_channel_layout_nb_channels(inlink->channel_layout);

    const int nch_out = s->nch;

    int ch;

    uint8_t *source_planes[MAX_CH];



    memcpy(source_planes, buf->extended_data,

           nch_in * sizeof(source_planes[0]));



    if (nch_out > nch_in) {

        if (nch_out > FF_ARRAY_ELEMS(buf->data)) {

            uint8_t **new_extended_data =

                av_mallocz(nch_out * sizeof(*buf->extended_data));

            if (!new_extended_data) {

                avfilter_unref_buffer(buf);

                return AVERROR(ENOMEM);

            }

            if (buf->extended_data == buf->data) {

                buf->extended_data = new_extended_data;

            } else {

                buf->extended_data = new_extended_data;

                av_free(buf->extended_data);

            }

        } else if (buf->extended_data != buf->data) {

            av_free(buf->extended_data);

            buf->extended_data = buf->data;

        }

    }



    for (ch = 0; ch < nch_out; ch++) {

        buf->extended_data[s->map[ch].out_channel_idx] =

            source_planes[s->map[ch].in_channel_idx];

    }



    if (buf->data != buf->extended_data)

        memcpy(buf->data, buf->extended_data,

           FFMIN(FF_ARRAY_ELEMS(buf->data), nch_out) * sizeof(buf->data[0]));



    return ff_filter_samples(outlink, buf);

}
