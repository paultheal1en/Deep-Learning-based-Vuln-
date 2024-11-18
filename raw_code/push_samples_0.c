static int push_samples(AVFilterLink *outlink)

{

    ASNSContext *asns = outlink->src->priv;

    AVFrame *outsamples = NULL;

    int ret, nb_out_samples, nb_pad_samples;



    if (asns->pad) {

        nb_out_samples = av_audio_fifo_size(asns->fifo) ? asns->nb_out_samples : 0;

        nb_pad_samples = nb_out_samples - FFMIN(nb_out_samples, av_audio_fifo_size(asns->fifo));

    } else {

        nb_out_samples = FFMIN(asns->nb_out_samples, av_audio_fifo_size(asns->fifo));

        nb_pad_samples = 0;

    }



    if (!nb_out_samples)

        return 0;



    outsamples = ff_get_audio_buffer(outlink, nb_out_samples);

    av_assert0(outsamples);



    av_audio_fifo_read(asns->fifo,

                       (void **)outsamples->extended_data, nb_out_samples);



    if (nb_pad_samples)

        av_samples_set_silence(outsamples->extended_data, nb_out_samples - nb_pad_samples,

                               nb_pad_samples, av_get_channel_layout_nb_channels(outlink->channel_layout),

                               outlink->format);

    outsamples->nb_samples     = nb_out_samples;

    outsamples->channel_layout = outlink->channel_layout;

    outsamples->sample_rate    = outlink->sample_rate;

    outsamples->pts = asns->next_out_pts;



    if (asns->next_out_pts != AV_NOPTS_VALUE)

        asns->next_out_pts += nb_out_samples;



    ret = ff_filter_frame(outlink, outsamples);

    if (ret < 0)

        return ret;

    asns->req_fullfilled = 1;

    return nb_out_samples;

}
