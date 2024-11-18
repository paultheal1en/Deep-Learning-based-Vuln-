int attribute_align_arg avcodec_decode_audio4(AVCodecContext *avctx,
                                              AVFrame *frame,
                                              int *got_frame_ptr,
                                              AVPacket *avpkt)
{
    int planar, channels;
    int ret = 0;
    *got_frame_ptr = 0;
    avctx->pkt = avpkt;
    if (!avpkt->data && avpkt->size) {
        av_log(avctx, AV_LOG_ERROR, "invalid packet: NULL data, size != 0\n");
        return AVERROR(EINVAL);
    }
    apply_param_change(avctx, avpkt);
    if ((avctx->codec->capabilities & CODEC_CAP_DELAY) || avpkt->size) {
        ret = avctx->codec->decode(avctx, frame, got_frame_ptr, avpkt);
        if (ret >= 0 && *got_frame_ptr) {
            avctx->frame_number++;
            frame->pkt_dts = avpkt->dts;
            if (frame->format == AV_SAMPLE_FMT_NONE)
                frame->format = avctx->sample_fmt;
        }
    }
    /* many decoders assign whole AVFrames, thus overwriting extended_data;
     * make sure it's set correctly; assume decoders that actually use
     * extended_data are doing it correctly */
    planar   = av_sample_fmt_is_planar(frame->format);
    channels = av_get_channel_layout_nb_channels(frame->channel_layout);
    if (!(planar && channels > AV_NUM_DATA_POINTERS))
        frame->extended_data = frame->data;
    return ret;
}