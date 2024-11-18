void ff_free_stream(AVFormatContext *s, AVStream *st){

    av_assert0(s->nb_streams>0);

    av_assert0(s->streams[ s->nb_streams-1 ] == st);



    if (st->codec) {

        avcodec_close(st->codec);

    }

    if (st->parser) {

        av_parser_close(st->parser);

    }

    if (st->attached_pic.data)

        av_free_packet(&st->attached_pic);

    av_dict_free(&st->metadata);

    av_freep(&st->probe_data.buf);

    av_freep(&st->index_entries);

    av_freep(&st->codec->extradata);

    av_freep(&st->codec->subtitle_header);

    av_freep(&st->codec);

    av_freep(&st->priv_data);

    if (st->info)

        av_freep(&st->info->duration_error);

    av_freep(&st->info);

    av_freep(&s->streams[ --s->nb_streams ]);

}
