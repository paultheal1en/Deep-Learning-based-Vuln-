static int chunk_mux_init(AVFormatContext *s)

{

    WebMChunkContext *wc = s->priv_data;

    AVFormatContext *oc;

    int ret;



    ret = avformat_alloc_output_context2(&wc->avf, wc->oformat, NULL, NULL);

    if (ret < 0)

        return ret;

    oc = wc->avf;



    oc->interrupt_callback = s->interrupt_callback;

    oc->max_delay          = s->max_delay;

    av_dict_copy(&oc->metadata, s->metadata, 0);



    oc->priv_data = av_mallocz(oc->oformat->priv_data_size);

    if (!oc->priv_data) {

        avio_close(oc->pb);

        return AVERROR(ENOMEM);

    }

    *(const AVClass**)oc->priv_data = oc->oformat->priv_class;

    av_opt_set_defaults(oc->priv_data);

    av_opt_set_int(oc->priv_data, "dash", 1, 0);

    av_opt_set_int(oc->priv_data, "cluster_time_limit", wc->chunk_duration, 0);

    av_opt_set_int(oc->priv_data, "live", 1, 0);



    oc->streams = s->streams;

    oc->nb_streams = s->nb_streams;



    return 0;

}
