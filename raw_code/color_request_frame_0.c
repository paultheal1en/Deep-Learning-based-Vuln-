static int color_request_frame(AVFilterLink *link)

{

    ColorContext *color = link->src->priv;

    AVFilterBufferRef *picref = ff_get_video_buffer(link, AV_PERM_WRITE, color->w, color->h);

    int ret;



    picref->video->pixel_aspect = (AVRational) {1, 1};

    picref->pts                 = color->pts++;

    picref->pos                 = -1;



    ret = ff_start_frame(link, avfilter_ref_buffer(picref, ~0));

    if (ret < 0)

        goto fail;



    ff_draw_rectangle(picref->data, picref->linesize,

                      color->line, color->line_step, color->hsub, color->vsub,

                      0, 0, color->w, color->h);

    ret = ff_draw_slice(link, 0, color->h, 1);

    if (ret < 0)

        goto fail;



    ret = ff_end_frame(link);



fail:

    avfilter_unref_buffer(picref);



    return ret;

}
