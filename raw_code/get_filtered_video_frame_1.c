int get_filtered_video_frame(AVFilterContext *ctx, AVFrame *frame,

                             AVFilterBufferRef **picref_ptr, AVRational *tb)

{

    int ret;

    AVFilterBufferRef *picref;




    if ((ret = avfilter_request_frame(ctx->inputs[0])) < 0)

        return ret;

    if (!(picref = ctx->inputs[0]->cur_buf))

        return AVERROR(ENOENT);

    *picref_ptr = picref;

    ctx->inputs[0]->cur_buf = NULL;

    *tb = ctx->inputs[0]->time_base;



    memcpy(frame->data,     picref->data,     sizeof(frame->data));

    memcpy(frame->linesize, picref->linesize, sizeof(frame->linesize));

    frame->pkt_pos          = picref->pos;

    frame->interlaced_frame = picref->video->interlaced;

    frame->top_field_first  = picref->video->top_field_first;

    frame->key_frame        = picref->video->key_frame;

    frame->pict_type        = picref->video->pict_type;

    frame->sample_aspect_ratio = picref->video->sample_aspect_ratio;



    return 1;

}