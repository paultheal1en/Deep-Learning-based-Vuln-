static AVFrame *hwmap_get_buffer(AVFilterLink *inlink, int w, int h)

{

    AVFilterContext *avctx = inlink->dst;

    AVFilterLink  *outlink = avctx->outputs[0];

    HWMapContext      *ctx = avctx->priv;



    if (ctx->map_backwards) {

        AVFrame *src, *dst;

        int err;



        src = ff_get_video_buffer(outlink, w, h);

        if (!src) {

            av_log(avctx, AV_LOG_ERROR, "Failed to allocate source "

                   "frame for software mapping.\n");

            return NULL;

        }



        dst = av_frame_alloc();

        if (!dst) {

            av_frame_free(&src);

            return NULL;

        }



        err = av_hwframe_map(dst, src, ctx->mode);

        if (err) {

            av_log(avctx, AV_LOG_ERROR, "Failed to map frame to "

                   "software: %d.\n", err);

            av_frame_free(&src);

            av_frame_free(&dst);

            return NULL;

        }



        av_frame_free(&src);

        return dst;

    } else {

        return ff_default_get_video_buffer(inlink, w, h);

    }

}
