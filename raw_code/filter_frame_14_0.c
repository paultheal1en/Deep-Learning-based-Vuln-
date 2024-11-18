static int filter_frame(AVFilterLink *inlink, AVFrame *buf)

{

    AVFilterContext *ctx = inlink->dst;

    AVFilterLink *outlink = ctx->outputs[0];

    DeflickerContext *s = ctx->priv;

    AVDictionary **metadata;

    AVFrame *out, *in;

    float f;

    int y;



    if (s->q.available < s->size && !s->eof) {

        s->luminance[s->available] = s->calc_avgy(ctx, buf);

        ff_bufqueue_add(ctx, &s->q, buf);

        s->available++;

        return 0;

    }



    in = ff_bufqueue_peek(&s->q, 0);



    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);

    if (!out) {

        av_frame_free(&buf);

        return AVERROR(ENOMEM);

    }



    s->get_factor(ctx, &f);

    s->deflicker(ctx, in->data[0], in->linesize[0], out->data[0], out->linesize[0],

                 outlink->w, outlink->h, f);

    for (y = 1; y < s->nb_planes; y++) {

        av_image_copy_plane(out->data[y], out->linesize[y],

                            in->data[y], in->linesize[y],

                            s->planewidth[y] * (1 + (s->depth > 8)), s->planeheight[y]);

    }



    av_frame_copy_props(out, in);

    metadata = &out->metadata;

    if (metadata) {

        uint8_t value[128];



        snprintf(value, sizeof(value), "%f", s->luminance[0]);

        av_dict_set(metadata, "lavfi.deflicker.luminance", value, 0);



        snprintf(value, sizeof(value), "%f", s->luminance[0] * f);

        av_dict_set(metadata, "lavfi.deflicker.new_luminance", value, 0);



        snprintf(value, sizeof(value), "%f", f - 1.0f);

        av_dict_set(metadata, "lavfi.deflicker.relative_change", value, 0);

    }



    in = ff_bufqueue_get(&s->q);

    av_frame_free(&in);

    memmove(&s->luminance[0], &s->luminance[1], sizeof(*s->luminance) * (s->size - 1));

    s->luminance[s->available - 1] = s->calc_avgy(ctx, buf);

    ff_bufqueue_add(ctx, &s->q, buf);



    return ff_filter_frame(outlink, out);

}
