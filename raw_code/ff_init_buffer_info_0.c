int ff_init_buffer_info(AVCodecContext *avctx, AVFrame *frame)

{

    if (avctx->internal->pkt) {

        frame->pkt_pts = avctx->internal->pkt->pts;

        av_frame_set_pkt_pos     (frame, avctx->internal->pkt->pos);

        av_frame_set_pkt_duration(frame, avctx->internal->pkt->duration);

        av_frame_set_pkt_size    (frame, avctx->internal->pkt->size);

    } else {

        frame->pkt_pts = AV_NOPTS_VALUE;

        av_frame_set_pkt_pos     (frame, -1);

        av_frame_set_pkt_duration(frame, 0);

        av_frame_set_pkt_size    (frame, -1);

    }

    frame->reordered_opaque = avctx->reordered_opaque;



    switch (avctx->codec->type) {

    case AVMEDIA_TYPE_VIDEO:

        if (frame->format < 0)

            frame->format              = avctx->pix_fmt;

        if (!frame->sample_aspect_ratio.num)

            frame->sample_aspect_ratio = avctx->sample_aspect_ratio;

        if (av_frame_get_colorspace(frame) == AVCOL_SPC_UNSPECIFIED)

            av_frame_set_colorspace(frame, avctx->colorspace);

        if (av_frame_get_color_range(frame) == AVCOL_RANGE_UNSPECIFIED)

            av_frame_set_color_range(frame, avctx->color_range);

        break;

    case AVMEDIA_TYPE_AUDIO:

        if (!frame->sample_rate)

            frame->sample_rate    = avctx->sample_rate;

        if (frame->format < 0)

            frame->format         = avctx->sample_fmt;

        if (!frame->channel_layout) {

            if (avctx->channel_layout) {

                 if (av_get_channel_layout_nb_channels(avctx->channel_layout) !=

                     avctx->channels) {

                     av_log(avctx, AV_LOG_ERROR, "Inconsistent channel "

                            "configuration.\n");

                     return AVERROR(EINVAL);

                 }



                frame->channel_layout = avctx->channel_layout;

            } else {

                if (avctx->channels > FF_SANE_NB_CHANNELS) {

                    av_log(avctx, AV_LOG_ERROR, "Too many channels: %d.\n",

                           avctx->channels);

                    return AVERROR(ENOSYS);

                }

            }

        }

        av_frame_set_channels(frame, avctx->channels);

        break;

    }

    return 0;

}
