static int filter_frame(AVFilterLink *link, AVFrame *frame)

{

    AVFilterContext *ctx = link->dst;

    AudioFIRContext *s = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    int ret = 0;



    av_audio_fifo_write(s->fifo[0], (void **)frame->extended_data,

                        frame->nb_samples);

    if (s->pts == AV_NOPTS_VALUE)

        s->pts = frame->pts;



    av_frame_free(&frame);



    if (!s->have_coeffs && s->eof_coeffs) {

        ret = convert_coeffs(ctx);

        if (ret < 0)

            return ret;

    }



    if (s->have_coeffs) {

        while (av_audio_fifo_size(s->fifo[0]) >= s->part_size) {

            ret = fir_frame(s, outlink);

            if (ret < 0)

                break;

        }

    }

    return ret;

}
