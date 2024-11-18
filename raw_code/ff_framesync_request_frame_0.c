int ff_framesync_request_frame(FFFrameSync *fs, AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    int input, ret;



    if ((ret = ff_framesync_process_frame(fs, 0)) < 0)

        return ret;

    if (ret > 0)

        return 0;

    if (fs->eof)

        return AVERROR_EOF;

    input = fs->in_request;

    ret = ff_request_frame(ctx->inputs[input]);

    if (ret == AVERROR_EOF) {

        if ((ret = ff_framesync_add_frame(fs, input, NULL)) < 0)

            return ret;

        if ((ret = ff_framesync_process_frame(fs, 0)) < 0)

            return ret;

        ret = 0;

    }

    return ret;

}
