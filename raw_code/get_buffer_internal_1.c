static int get_buffer_internal(AVCodecContext *avctx, AVFrame *frame, int flags)
{
    const AVHWAccel *hwaccel = avctx->hwaccel;
    int override_dimensions = 1;
    int ret;
    if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) {
        if ((ret = av_image_check_size2(avctx->width, avctx->height, avctx->max_pixels, AV_PIX_FMT_NONE, 0, avctx)) < 0 || avctx->pix_fmt<0) {
            av_log(avctx, AV_LOG_ERROR, "video_get_buffer: image parameters invalid\n");
            return AVERROR(EINVAL);
        }
        if (frame->width <= 0 || frame->height <= 0) {
            frame->width  = FFMAX(avctx->width,  AV_CEIL_RSHIFT(avctx->coded_width,  avctx->lowres));
            frame->height = FFMAX(avctx->height, AV_CEIL_RSHIFT(avctx->coded_height, avctx->lowres));
            override_dimensions = 0;
        }
        if (frame->data[0] || frame->data[1] || frame->data[2] || frame->data[3]) {
            av_log(avctx, AV_LOG_ERROR, "pic->data[*]!=NULL in get_buffer_internal\n");
            return AVERROR(EINVAL);
        }
    }
    ret = ff_decode_frame_props(avctx, frame);
        return ret;
    if (hwaccel) {
        if (hwaccel->alloc_frame) {
            ret = hwaccel->alloc_frame(avctx, frame);
            goto end;
        }
    } else
        avctx->sw_pix_fmt = avctx->pix_fmt;
    ret = avctx->get_buffer2(avctx, frame, flags);
    if (ret >= 0)
        validate_avframe_allocation(avctx, frame);
end:
    if (avctx->codec_type == AVMEDIA_TYPE_VIDEO && !override_dimensions &&
        !(avctx->codec->caps_internal & FF_CODEC_CAP_EXPORTS_CROPPING)) {
        frame->width  = avctx->width;
        frame->height = avctx->height;
    }
    return ret;
}