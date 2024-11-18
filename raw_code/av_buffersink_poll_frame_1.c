int av_buffersink_poll_frame(AVFilterContext *ctx)

{

    BufferSinkContext *buf = ctx->priv;

    AVFilterLink *inlink = ctx->inputs[0];



    av_assert0(!strcmp(ctx->filter->name, "buffersink") || !strcmp(ctx->filter->name, "abuffersink"));



    return av_fifo_size(buf->fifo)/sizeof(AVFilterBufferRef *) + ff_poll_frame(inlink);

}
