static int nsv_read_close(AVFormatContext *s)

{

/*     int i; */

    NSVContext *nsv = s->priv_data;



    av_freep(&nsv->nsvs_file_offset);

    av_freep(&nsv->nsvs_timestamps);







#if 0



    for(i=0;i<s->nb_streams;i++) {

        AVStream *st = s->streams[i];

        NSVStream *ast = st->priv_data;

        if(ast){

            av_free(ast->index_entries);

            av_free(ast);

        }

        av_free(st->codec->palctrl);

    }



#endif

    return 0;

}