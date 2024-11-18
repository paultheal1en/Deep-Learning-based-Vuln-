void avformat_free_context(AVFormatContext *s)

{

    int i;



    if (!s)

        return;



    av_opt_free(s);

    if (s->iformat && s->iformat->priv_class && s->priv_data)

        av_opt_free(s->priv_data);

    if (s->oformat && s->oformat->priv_class && s->priv_data)

        av_opt_free(s->priv_data);



    for (i = s->nb_streams - 1; i >= 0; i--) {

        ff_free_stream(s, s->streams[i]);

    }

    for (i = s->nb_programs - 1; i >= 0; i--) {

        av_dict_free(&s->programs[i]->metadata);

        av_freep(&s->programs[i]->stream_index);

        av_freep(&s->programs[i]);

    }

    av_freep(&s->programs);

    av_freep(&s->priv_data);

    while (s->nb_chapters--) {

        av_dict_free(&s->chapters[s->nb_chapters]->metadata);

        av_freep(&s->chapters[s->nb_chapters]);

    }

    av_freep(&s->chapters);

    av_dict_free(&s->metadata);

    av_freep(&s->streams);

    av_freep(&s->internal);


    av_free(s);

}