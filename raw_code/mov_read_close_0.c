static int mov_read_close(AVFormatContext *s)

{

    MOVContext *mov = s->priv_data;

    int i, j;



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MOVStreamContext *sc = st->priv_data;



        av_freep(&sc->ctts_data);

        for (j = 0; j < sc->drefs_count; j++) {

            av_freep(&sc->drefs[j].path);

            av_freep(&sc->drefs[j].dir);

        }

        av_freep(&sc->drefs);

        if (sc->pb && sc->pb != s->pb)

            avio_close(sc->pb);



        av_freep(&sc->chunk_offsets);

        av_freep(&sc->stsc_data);

        av_freep(&sc->sample_sizes);

        av_freep(&sc->keyframes);

        av_freep(&sc->stts_data);

        av_freep(&sc->stps_data);

        av_freep(&sc->rap_group);

        av_freep(&sc->display_matrix);

    }



    if (mov->dv_demux) {

        avformat_free_context(mov->dv_fctx);

        mov->dv_fctx = NULL;

    }



    av_freep(&mov->trex_data);



    return 0;

}
