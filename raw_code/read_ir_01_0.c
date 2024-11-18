static int read_ir(AVFilterLink *link, AVFrame *frame)

{

    AVFilterContext *ctx = link->dst;

    AudioFIRContext *s = ctx->priv;

    int nb_taps, max_nb_taps;



    av_audio_fifo_write(s->fifo[1], (void **)frame->extended_data,

                        frame->nb_samples);

    av_frame_free(&frame);



    nb_taps = av_audio_fifo_size(s->fifo[1]);

    max_nb_taps = MAX_IR_DURATION * ctx->outputs[0]->sample_rate;

    if (nb_taps > max_nb_taps) {

        av_log(ctx, AV_LOG_ERROR, "Too big number of coefficients: %d > %d.\n", nb_taps, max_nb_taps);

        return AVERROR(EINVAL);

    }



    return 0;

}
