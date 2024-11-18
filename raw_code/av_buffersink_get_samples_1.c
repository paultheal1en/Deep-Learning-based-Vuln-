int av_buffersink_get_samples(AVFilterContext *ctx, AVFrame *frame, int nb_samples)

{

    BufferSinkContext *s = ctx->priv;

    AVFilterLink   *link = ctx->inputs[0];

    AVFrame *cur_frame;

    int ret = 0;



    if (!s->audio_fifo) {

        int nb_channels = link->channels;

        if (!(s->audio_fifo = av_audio_fifo_alloc(link->format, nb_channels, nb_samples)))

            return AVERROR(ENOMEM);

    }



    while (ret >= 0) {

        if (av_audio_fifo_size(s->audio_fifo) >= nb_samples)

            return read_from_fifo(ctx, frame, nb_samples);



        if (!(cur_frame = av_frame_alloc()))

            return AVERROR(ENOMEM);

        ret = av_buffersink_get_frame_flags(ctx, cur_frame, 0);

        if (ret == AVERROR_EOF && av_audio_fifo_size(s->audio_fifo)) {

            av_frame_free(&cur_frame);

            return read_from_fifo(ctx, frame, av_audio_fifo_size(s->audio_fifo));

        } else if (ret < 0) {

            av_frame_free(&cur_frame);

            return ret;

        }



        if (cur_frame->pts != AV_NOPTS_VALUE) {

            s->next_pts = cur_frame->pts -

                          av_rescale_q(av_audio_fifo_size(s->audio_fifo),

                                       (AVRational){ 1, link->sample_rate },

                                       link->time_base);

        }



        ret = av_audio_fifo_write(s->audio_fifo, (void**)cur_frame->extended_data,

                                  cur_frame->nb_samples);

        av_frame_free(&cur_frame);

    }



    return ret;



}
