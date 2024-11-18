static int update_streams_from_subdemuxer(AVFormatContext *s, struct playlist *pls)

{

    while (pls->n_main_streams < pls->ctx->nb_streams) {

        int ist_idx = pls->n_main_streams;

        AVStream *st = avformat_new_stream(s, NULL);

        AVStream *ist = pls->ctx->streams[ist_idx];



        if (!st)

            return AVERROR(ENOMEM);



        st->id = pls->index;

        set_stream_info_from_input_stream(st, pls, ist);



        dynarray_add(&pls->main_streams, &pls->n_main_streams, st);



        add_stream_to_programs(s, pls, st);

    }



    return 0;

}
