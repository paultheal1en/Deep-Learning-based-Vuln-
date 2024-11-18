static int xmv_read_close(AVFormatContext *s)

{

    XMVDemuxContext *xmv = s->priv_data;



    av_free(xmv->audio);

    av_free(xmv->audio_tracks);



    return 0;

}
