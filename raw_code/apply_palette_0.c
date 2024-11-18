static AVFrame *apply_palette(AVFilterLink *inlink, AVFrame *in)

{

    int x, y, w, h;

    AVFilterContext *ctx = inlink->dst;

    PaletteUseContext *s = ctx->priv;

    AVFilterLink *outlink = inlink->dst->outputs[0];



    AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);

    if (!out) {

        av_frame_free(&in);

        return NULL;

    }

    av_frame_copy_props(out, in);



    set_processing_window(s->diff_mode, s->last_in, in,

                          s->last_out, out, &x, &y, &w, &h);

    av_frame_free(&s->last_in);

    av_frame_free(&s->last_out);

    s->last_in  = av_frame_clone(in);

    s->last_out = av_frame_clone(out);

    if (!s->last_in || !s->last_out ||

        av_frame_make_writable(s->last_in) < 0) {

        av_frame_free(&in);

        av_frame_free(&out);

        return NULL;

    }



    ff_dlog(ctx, "%dx%d rect: (%d;%d) -> (%d,%d) [area:%dx%d]\n",

            w, h, x, y, x+w, y+h, in->width, in->height);



    if (s->set_frame(s, out, in, x, y, w, h) < 0) {

        av_frame_free(&out);

        return NULL;

    }

    memcpy(out->data[1], s->palette, AVPALETTE_SIZE);

    if (s->calc_mean_err)

        debug_mean_error(s, in, out, inlink->frame_count_out);

    av_frame_free(&in);

    return out;

}
