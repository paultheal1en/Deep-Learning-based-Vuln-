static av_cold int asink_init(AVFilterContext *ctx, const char *args, void *opaque)

{

    BufferSinkContext *buf = ctx->priv;

    AVABufferSinkParams *params;



    if (!opaque) {

        av_log(ctx, AV_LOG_ERROR,

               "No opaque field provided, an AVABufferSinkParams struct is required\n");

        return AVERROR(EINVAL);

    } else

        params = (AVABufferSinkParams *)opaque;



    buf->sample_fmts     = params->sample_fmts;

    buf->channel_layouts = params->channel_layouts;

    buf->packing_fmts    = params->packing_fmts;



    return common_init(ctx);

}
