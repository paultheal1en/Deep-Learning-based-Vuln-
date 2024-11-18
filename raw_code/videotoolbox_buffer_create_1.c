static int videotoolbox_buffer_create(AVCodecContext *avctx, AVFrame *frame)

{

    VTContext *vtctx = avctx->internal->hwaccel_priv_data;

    CVPixelBufferRef pixbuf = (CVPixelBufferRef)vtctx->frame;

    OSType pixel_format = CVPixelBufferGetPixelFormatType(pixbuf);

    enum AVPixelFormat sw_format = av_map_videotoolbox_format_to_pixfmt(pixel_format);

    int width = CVPixelBufferGetWidth(pixbuf);

    int height = CVPixelBufferGetHeight(pixbuf);

    AVHWFramesContext *cached_frames;

    int ret;



    ret = ff_videotoolbox_buffer_create(vtctx, frame);

    if (ret < 0)

        return ret;



    // Old API code path.

    if (!vtctx->cached_hw_frames_ctx)

        return 0;



    cached_frames = (AVHWFramesContext*)vtctx->cached_hw_frames_ctx->data;



    if (cached_frames->sw_format != sw_format ||

        cached_frames->width != width ||

        cached_frames->height != height) {

        AVBufferRef *hw_frames_ctx = av_hwframe_ctx_alloc(cached_frames->device_ref);

        AVHWFramesContext *hw_frames;

        if (!hw_frames_ctx)

            return AVERROR(ENOMEM);



        hw_frames = (AVHWFramesContext*)hw_frames_ctx->data;

        hw_frames->format = cached_frames->format;

        hw_frames->sw_format = sw_format;

        hw_frames->width = width;

        hw_frames->height = height;



        ret = av_hwframe_ctx_init(hw_frames_ctx);

        if (ret < 0) {

            av_buffer_unref(&hw_frames_ctx);

            return ret;

        }



        av_buffer_unref(&vtctx->cached_hw_frames_ctx);

        vtctx->cached_hw_frames_ctx = hw_frames_ctx;

    }



    av_assert0(!frame->hw_frames_ctx);

    frame->hw_frames_ctx = av_buffer_ref(vtctx->cached_hw_frames_ctx);

    if (!frame->hw_frames_ctx)

        return AVERROR(ENOMEM);



    return 0;

}
