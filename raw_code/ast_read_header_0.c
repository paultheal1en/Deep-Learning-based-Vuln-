static int ast_read_header(AVFormatContext *s)

{

    int codec;

    AVStream *st;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    avio_skip(s->pb, 8);

    codec = avio_rb16(s->pb);

    switch (codec) {

    case 1:

        st->codec->codec_id = AV_CODEC_ID_PCM_S16BE_PLANAR;

        break;

    default:

        av_log(s, AV_LOG_ERROR, "unsupported codec %d\n", codec);

    }



    avio_skip(s->pb, 2);



    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

    st->codec->channels = avio_rb16(s->pb);

    if (!st->codec->channels)

        return AVERROR_INVALIDDATA;



    if (st->codec->channels == 2)

        st->codec->channel_layout = AV_CH_LAYOUT_STEREO;

    else if (st->codec->channels == 4)

        st->codec->channel_layout = AV_CH_LAYOUT_4POINT0;



    avio_skip(s->pb, 2);

    st->codec->sample_rate = avio_rb32(s->pb);

    if (st->codec->sample_rate <= 0)

        return AVERROR_INVALIDDATA;

    st->start_time         = 0;

    st->duration           = avio_rb32(s->pb);

    avio_skip(s->pb, 40);

    avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);



    return 0;

}
