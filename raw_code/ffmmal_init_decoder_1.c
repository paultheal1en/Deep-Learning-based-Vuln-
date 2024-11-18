static av_cold int ffmmal_init_decoder(AVCodecContext *avctx)
{
    MMALDecodeContext *ctx = avctx->priv_data;
    MMAL_STATUS_T status;
    MMAL_ES_FORMAT_T *format_in;
    MMAL_COMPONENT_T *decoder;
    char tmp[32];
    int ret = 0;
    bcm_host_init();
    if (mmal_vc_init()) {
        av_log(avctx, AV_LOG_ERROR, "Cannot initialize MMAL VC driver!\n");
        return AVERROR(ENOSYS);
    if ((ret = ff_get_format(avctx, avctx->codec->pix_fmts)) < 0)
        return ret;
    avctx->pix_fmt = ret;
    if ((status = mmal_component_create(MMAL_COMPONENT_DEFAULT_VIDEO_DECODER, &ctx->decoder)))
        goto fail;
    decoder = ctx->decoder;
    format_in = decoder->input[0]->format;
    format_in->type = MMAL_ES_TYPE_VIDEO;
    switch (avctx->codec_id) {
        case AV_CODEC_ID_MPEG2VIDEO:
            format_in->encoding = MMAL_ENCODING_MP2V;
            break;
        case AV_CODEC_ID_MPEG4:
            format_in->encoding = MMAL_ENCODING_MP4V;
            break;
        case AV_CODEC_ID_VC1:
            format_in->encoding = MMAL_ENCODING_WVC1;
            break;
        case AV_CODEC_ID_H264:
        default:
            format_in->encoding = MMAL_ENCODING_H264;
            break;
    format_in->es->video.width = FFALIGN(avctx->width, 32);
    format_in->es->video.height = FFALIGN(avctx->height, 16);
    format_in->es->video.crop.width = avctx->width;
    format_in->es->video.crop.height = avctx->height;
    format_in->es->video.frame_rate.num = 24000;
    format_in->es->video.frame_rate.den = 1001;
    format_in->es->video.par.num = avctx->sample_aspect_ratio.num;
    format_in->es->video.par.den = avctx->sample_aspect_ratio.den;
    format_in->flags = MMAL_ES_FORMAT_FLAG_FRAMED;
    av_get_codec_tag_string(tmp, sizeof(tmp), format_in->encoding);
    av_log(avctx, AV_LOG_DEBUG, "Using MMAL %s encoding.\n", tmp);
    if ((status = mmal_port_format_commit(decoder->input[0])))
        goto fail;
    decoder->input[0]->buffer_num =
        FFMAX(decoder->input[0]->buffer_num_min, 20);
    decoder->input[0]->buffer_size =
        FFMAX(decoder->input[0]->buffer_size_min, 512 * 1024);
    ctx->pool_in = mmal_pool_create(decoder->input[0]->buffer_num, 0);
    if (!ctx->pool_in) {
        ret = AVERROR(ENOMEM);
        goto fail;
    if ((ret = ffmal_update_format(avctx)) < 0)
        goto fail;
    ctx->queue_decoded_frames = mmal_queue_create();
    if (!ctx->queue_decoded_frames)
        goto fail;
    decoder->input[0]->userdata = (void*)avctx;
    decoder->output[0]->userdata = (void*)avctx;
    decoder->control->userdata = (void*)avctx;
    if ((status = mmal_port_enable(decoder->control, control_port_cb)))
        goto fail;
    if ((status = mmal_port_enable(decoder->input[0], input_callback)))
        goto fail;
    if ((status = mmal_port_enable(decoder->output[0], output_callback)))
        goto fail;
    if ((status = mmal_component_enable(decoder)))
        goto fail;
    return 0;
fail:
    ffmmal_close_decoder(avctx);
    return ret < 0 ? ret : AVERROR_UNKNOWN;