static int nut_read_close(AVFormatContext *s)

{

    NUTContext *nut = s->priv_data;




    av_freep(&nut->time_base);

    av_freep(&nut->stream);






    return 0;

}