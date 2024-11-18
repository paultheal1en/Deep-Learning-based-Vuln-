static int hls_window(AVFormatContext *s, int last)

{

    HLSContext *hls = s->priv_data;

    ListEntry *en;

    int64_t target_duration = 0;

    int ret = 0;

    AVIOContext *out = NULL;

    char temp_filename[1024];

    int64_t sequence = FFMAX(hls->start_sequence, hls->sequence - hls->size);



    snprintf(temp_filename, sizeof(temp_filename), "%s.tmp", s->filename);

    if ((ret = avio_open2(&out, temp_filename, AVIO_FLAG_WRITE,

                          &s->interrupt_callback, NULL)) < 0)

        goto fail;



    for (en = hls->list; en; en = en->next) {

        if (target_duration < en->duration)

            target_duration = en->duration;

    }



    avio_printf(out, "#EXTM3U\n");

    avio_printf(out, "#EXT-X-VERSION:%d\n", hls->version);

    if (hls->allowcache == 0 || hls->allowcache == 1) {

        avio_printf(out, "#EXT-X-ALLOW-CACHE:%s\n", hls->allowcache == 0 ? "NO" : "YES");

    }

    avio_printf(out, "#EXT-X-TARGETDURATION:%"PRId64"\n",

                av_rescale_rnd(target_duration, 1, AV_TIME_BASE,

                               AV_ROUND_UP));

    avio_printf(out, "#EXT-X-MEDIA-SEQUENCE:%"PRId64"\n", sequence);



    av_log(s, AV_LOG_VERBOSE, "EXT-X-MEDIA-SEQUENCE:%"PRId64"\n",

           sequence);



    for (en = hls->list; en; en = en->next) {

        if (hls->version > 2)

            avio_printf(out, "#EXTINF:%f\n",

                        (double)en->duration / AV_TIME_BASE);

        else

            avio_printf(out, "#EXTINF:%"PRId64",\n",

                        av_rescale(en->duration, 1, AV_TIME_BASE));

        if (hls->baseurl)

            avio_printf(out, "%s", hls->baseurl);

        avio_printf(out, "%s\n", en->name);

    }



    if (last)

        avio_printf(out, "#EXT-X-ENDLIST\n");



fail:

    avio_closep(&out);

    if (ret >= 0)

        ff_rename(temp_filename, s->filename);

    return ret;

}
