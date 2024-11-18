static int wav_parse_fmt_tag(AVFormatContext *s, int64_t size, AVStream **st)

{

    AVIOContext *pb = s->pb;

    int ret;



    /* parse fmt header */

    *st = av_new_stream(s, 0);

    if (!*st)

        return AVERROR(ENOMEM);



    ff_get_wav_header(pb, (*st)->codec, size);

    if (ret < 0)

        return ret;

    (*st)->need_parsing = AVSTREAM_PARSE_FULL;



    av_set_pts_info(*st, 64, 1, (*st)->codec->sample_rate);



    return 0;

}
