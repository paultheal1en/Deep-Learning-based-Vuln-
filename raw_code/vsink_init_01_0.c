static av_cold int vsink_init(AVFilterContext *ctx, void *opaque)

{

    BufferSinkContext *buf = ctx->priv;

    AVBufferSinkParams *params = opaque;



    if (params && params->pixel_fmts) {

        const int *pixel_fmts = params->pixel_fmts;



        buf->pixel_fmts = ff_copy_int_list(pixel_fmts);

        if (!buf->pixel_fmts)

            return AVERROR(ENOMEM);

    }



    return common_init(ctx);

}
