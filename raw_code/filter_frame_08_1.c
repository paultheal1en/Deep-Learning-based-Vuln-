static int filter_frame(AVFilterLink *inlink, AVFrame *insamplesref)

{

    AResampleContext *aresample = inlink->dst->priv;

    const int n_in  = insamplesref->nb_samples;

    int64_t delay;

    int n_out       = n_in * aresample->ratio + 32;

    AVFilterLink *const outlink = inlink->dst->outputs[0];

    AVFrame *outsamplesref;

    int ret;



    delay = swr_get_delay(aresample->swr, outlink->sample_rate);

    if (delay > 0)

        n_out += FFMIN(delay, FFMAX(4096, n_out));



    outsamplesref = ff_get_audio_buffer(outlink, n_out);



    if(!outsamplesref)

        return AVERROR(ENOMEM);



    av_frame_copy_props(outsamplesref, insamplesref);

    outsamplesref->format                = outlink->format;

    outsamplesref->channels              = outlink->channels;

    outsamplesref->channel_layout        = outlink->channel_layout;

    outsamplesref->sample_rate           = outlink->sample_rate;



    if(insamplesref->pts != AV_NOPTS_VALUE) {

        int64_t inpts = av_rescale(insamplesref->pts, inlink->time_base.num * (int64_t)outlink->sample_rate * inlink->sample_rate, inlink->time_base.den);

        int64_t outpts= swr_next_pts(aresample->swr, inpts);

        aresample->next_pts =

        outsamplesref->pts  = ROUNDED_DIV(outpts, inlink->sample_rate);

    } else {

        outsamplesref->pts  = AV_NOPTS_VALUE;

    }

    n_out = swr_convert(aresample->swr, outsamplesref->extended_data, n_out,

                                 (void *)insamplesref->extended_data, n_in);

    if (n_out <= 0) {

        av_frame_free(&outsamplesref);

        av_frame_free(&insamplesref);

        return 0;

    }



    aresample->more_data = outsamplesref->nb_samples == n_out; // Indicate that there is probably more data in our buffers



    outsamplesref->nb_samples  = n_out;



    ret = ff_filter_frame(outlink, outsamplesref);

    av_frame_free(&insamplesref);

    return ret;

}
