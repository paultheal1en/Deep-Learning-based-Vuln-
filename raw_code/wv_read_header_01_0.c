static int wv_read_header(AVFormatContext *s,

                          AVFormatParameters *ap)

{

    ByteIOContext *pb = s->pb;

    WVContext *wc = s->priv_data;

    AVStream *st;



    if(wv_read_block_header(s, pb) < 0)

        return -1;



    wc->block_parsed = 0;

    /* now we are ready: build format streams */

    st = av_new_stream(s, 0);

    if (!st)

        return -1;

    st->codec->codec_type = CODEC_TYPE_AUDIO;

    st->codec->codec_id = CODEC_ID_WAVPACK;

    st->codec->channels = wc->chan;

    st->codec->sample_rate = wc->rate;

    st->codec->bits_per_coded_sample = wc->bpp;

    av_set_pts_info(st, 64, 1, wc->rate);

    s->start_time = 0;

    s->duration = (int64_t)wc->samples * AV_TIME_BASE / st->codec->sample_rate;



    if(!url_is_streamed(s->pb)) {

        int64_t cur = url_ftell(s->pb);

        ff_ape_parse_tag(s);

        if(!av_metadata_get(s->metadata, "", NULL, AV_METADATA_IGNORE_SUFFIX))

            ff_id3v1_read(s);

        url_fseek(s->pb, cur, SEEK_SET);

    }



    return 0;

}
