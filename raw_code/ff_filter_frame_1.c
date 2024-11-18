int ff_filter_frame(AVFilterLink *link, AVFrame *frame)

{

    int (*filter_frame)(AVFilterLink *, AVFrame *);

    AVFilterPad *dst = link->dstpad;

    AVFrame *out;



    FF_DPRINTF_START(NULL, filter_frame);

    ff_dlog_link(NULL, link, 1);



    if (!(filter_frame = dst->filter_frame))

        filter_frame = default_filter_frame;



    /* copy the frame if needed */

    if (dst->needs_writable && !av_frame_is_writable(frame)) {

        av_log(link->dst, AV_LOG_DEBUG, "Copying data in avfilter.\n");



        switch (link->type) {

        case AVMEDIA_TYPE_VIDEO:

            out = ff_get_video_buffer(link, link->w, link->h);

            break;

        case AVMEDIA_TYPE_AUDIO:

            out = ff_get_audio_buffer(link, frame->nb_samples);

            break;

        default: return AVERROR(EINVAL);

        }

        if (!out) {

            av_frame_free(&frame);

            return AVERROR(ENOMEM);

        }

        av_frame_copy_props(out, frame);



        switch (link->type) {

        case AVMEDIA_TYPE_VIDEO:

            av_image_copy(out->data, out->linesize, frame->data, frame->linesize,

                          frame->format, frame->width, frame->height);

            break;

        case AVMEDIA_TYPE_AUDIO:

            av_samples_copy(out->extended_data, frame->extended_data,

                            0, 0, frame->nb_samples,

                            av_get_channel_layout_nb_channels(frame->channel_layout),

                            frame->format);

            break;

        default: return AVERROR(EINVAL);

        }



        av_frame_free(&frame);

    } else

        out = frame;



    return filter_frame(link, out);

}
