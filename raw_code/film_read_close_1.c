static int film_read_close(AVFormatContext *s)

{

    FilmDemuxContext *film = s->priv_data;



    av_free(film->sample_table);

    av_free(film->stereo_buffer);



    return 0;

}
