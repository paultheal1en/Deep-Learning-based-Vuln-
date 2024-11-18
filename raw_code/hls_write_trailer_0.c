static int hls_write_trailer(struct AVFormatContext *s)

{

    HLSContext *hls = s->priv_data;

    AVFormatContext *oc = hls->avf;



    av_write_trailer(oc);

    avio_closep(&oc->pb);

    avformat_free_context(oc);

    av_free(hls->basename);

    append_entry(hls, hls->duration);

    hls_window(s, 1);



    free_entries(hls);

    return 0;

}
