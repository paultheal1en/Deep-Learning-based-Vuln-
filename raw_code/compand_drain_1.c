static int compand_drain(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    CompandContext *s    = ctx->priv;

    const int channels   = outlink->channels;

    AVFrame *frame       = NULL;

    int chan, i, dindex;



    /* 2048 is to limit output frame size during drain */

    frame = ff_get_audio_buffer(outlink, FFMIN(2048, s->delay_count));

    if (!frame)

        return AVERROR(ENOMEM);

    frame->pts = s->pts;

    s->pts += av_rescale_q(frame->nb_samples,

            (AVRational){ 1, outlink->sample_rate }, outlink->time_base);




    for (chan = 0; chan < channels; chan++) {

        AVFrame *delay_frame = s->delay_frame;

        double *dbuf = (double *)delay_frame->extended_data[chan];

        double *dst = (double *)frame->extended_data[chan];

        ChanParam *cp = &s->channels[chan];



        dindex = s->delay_index;

        for (i = 0; i < frame->nb_samples; i++) {

            dst[i] = av_clipd(dbuf[dindex] * get_volume(s, cp->volume),

                    -1, 1);

            dindex = MOD(dindex + 1, s->delay_samples);

        }

    }

    s->delay_count -= frame->nb_samples;

    s->delay_index = dindex;



    return ff_filter_frame(outlink, frame);

}