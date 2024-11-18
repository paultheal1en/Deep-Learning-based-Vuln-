int ff_raw_video_read_header(AVFormatContext *s)

{

    AVStream *st;

    FFRawVideoDemuxerContext *s1 = s->priv_data;

    AVRational framerate;

    int ret = 0;





    st = avformat_new_stream(s, NULL);

    if (!st) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = s->iformat->raw_codec_id;

    st->need_parsing = AVSTREAM_PARSE_FULL;



    if ((ret = av_parse_video_rate(&framerate, s1->framerate)) < 0) {

        av_log(s, AV_LOG_ERROR, "Could not parse framerate: %s.\n", s1->framerate);

        goto fail;

    }



    st->r_frame_rate = st->avg_frame_rate = framerate;

    avpriv_set_pts_info(st, 64, framerate.den, framerate.num);



fail:

    return ret;

}
