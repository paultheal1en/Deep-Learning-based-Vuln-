int attribute_align_arg avcodec_encode_audio2(AVCodecContext *avctx,
                                              AVPacket *avpkt,
                                              const AVFrame *frame,
                                              int *got_packet_ptr)
{
    AVFrame *extended_frame = NULL;
    AVFrame *padded_frame = NULL;
    int ret;
    AVPacket user_pkt = *avpkt;
    int needs_realloc = !user_pkt.data;
    *got_packet_ptr = 0;
    if (!(avctx->codec->capabilities & AV_CODEC_CAP_DELAY) && !frame) {
        av_free_packet(avpkt);
        av_init_packet(avpkt);
        return 0;
    }
    /* ensure that extended_data is properly set */
    if (frame && !frame->extended_data) {
        if (av_sample_fmt_is_planar(avctx->sample_fmt) &&
            avctx->channels > AV_NUM_DATA_POINTERS) {
            av_log(avctx, AV_LOG_ERROR, "Encoding to a planar sample format, "
                                        "with more than %d channels, but extended_data is not set.\n",
                   AV_NUM_DATA_POINTERS);
            return AVERROR(EINVAL);
        }
        av_log(avctx, AV_LOG_WARNING, "extended_data is not set.\n");
        extended_frame = av_frame_alloc();
        if (!extended_frame)
            return AVERROR(ENOMEM);
        memcpy(extended_frame, frame, sizeof(AVFrame));
        extended_frame->extended_data = extended_frame->data;
        frame = extended_frame;
    }
    /* extract audio service type metadata */
    if (frame) {
        AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_AUDIO_SERVICE_TYPE);
        if (sd && sd->size >= sizeof(enum AVAudioServiceType))
            avctx->audio_service_type = *(enum AVAudioServiceType*)sd->data;
    }
    /* check for valid frame size */
    if (frame) {
        if (avctx->codec->capabilities & AV_CODEC_CAP_SMALL_LAST_FRAME) {
            if (frame->nb_samples > avctx->frame_size) {
                av_log(avctx, AV_LOG_ERROR, "more samples than frame size (avcodec_encode_audio2)\n");
                ret = AVERROR(EINVAL);
                goto end;
            }
        } else if (!(avctx->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)) {
            if (frame->nb_samples < avctx->frame_size &&
                !avctx->internal->last_audio_frame) {
                ret = pad_last_frame(avctx, &padded_frame, frame);
                if (ret < 0)
                    goto end;
                frame = padded_frame;
                avctx->internal->last_audio_frame = 1;
            }
            if (frame->nb_samples != avctx->frame_size) {
                av_log(avctx, AV_LOG_ERROR, "nb_samples (%d) != frame_size (%d) (avcodec_encode_audio2)\n", frame->nb_samples, avctx->frame_size);
                ret = AVERROR(EINVAL);
                goto end;
            }
        }
    }
    ret = avctx->codec->encode2(avctx, avpkt, frame, got_packet_ptr);
    if (!ret) {
        if (*got_packet_ptr) {
            if (!(avctx->codec->capabilities & AV_CODEC_CAP_DELAY)) {
                if (avpkt->pts == AV_NOPTS_VALUE)
                    avpkt->pts = frame->pts;
                if (!avpkt->duration)
                    avpkt->duration = ff_samples_to_time_base(avctx,
                                                              frame->nb_samples);
            }
            avpkt->dts = avpkt->pts;
        } else {
            avpkt->size = 0;
        }
    }
    if (avpkt->data && avpkt->data == avctx->internal->byte_buffer) {
        needs_realloc = 0;
        if (user_pkt.data) {
            if (user_pkt.size >= avpkt->size) {
                memcpy(user_pkt.data, avpkt->data, avpkt->size);
            } else {
                av_log(avctx, AV_LOG_ERROR, "Provided packet is too small, needs to be %d\n", avpkt->size);
                avpkt->size = user_pkt.size;
                ret = -1;
            }
            avpkt->buf      = user_pkt.buf;
            avpkt->data     = user_pkt.data;
#if FF_API_DESTRUCT_PACKET
FF_DISABLE_DEPRECATION_WARNINGS
            avpkt->destruct = user_pkt.destruct;
FF_ENABLE_DEPRECATION_WARNINGS
#endif
        } else {
            if (av_dup_packet(avpkt) < 0) {
                ret = AVERROR(ENOMEM);
            }
        }
    }
    if (!ret) {
        if (needs_realloc && avpkt->data) {
            ret = av_buffer_realloc(&avpkt->buf, avpkt->size + AV_INPUT_BUFFER_PADDING_SIZE);
            if (ret >= 0)
                avpkt->data = avpkt->buf->data;
        }
        avctx->frame_number++;
    }
    if (ret < 0 || !*got_packet_ptr) {
        av_free_packet(avpkt);
        av_init_packet(avpkt);
        goto end;
    }
    /* NOTE: if we add any audio encoders which output non-keyframe packets,
     *       this needs to be moved to the encoders, but for now we can do it
     *       here to simplify things */
    avpkt->flags |= AV_PKT_FLAG_KEY;
end:
    av_frame_free(&padded_frame);
    av_free(extended_frame);
#if FF_API_AUDIOENC_DELAY
    avctx->delay = avctx->initial_padding;
#endif
    return ret;
}