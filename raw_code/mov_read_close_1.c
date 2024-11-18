static int mov_read_close(AVFormatContext *s)

{

    int i;

    MOVContext *mov = s->priv_data;

    for(i=0; i<mov->total_streams; i++)

        mov_free_stream_context(mov->streams[i]);

    for(i=0; i<s->nb_streams; i++)

        av_free(s->streams[i]);

    return 0;

}
