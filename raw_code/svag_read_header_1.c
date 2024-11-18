static int svag_read_header(AVFormatContext *s)

{

    unsigned size, align;

    AVStream *st;



    avio_skip(s->pb, 4);



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    size                   = avio_rl32(s->pb);

    st->codec->codec_type  = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id    = AV_CODEC_ID_ADPCM_PSX;

    st->codec->sample_rate = avio_rl32(s->pb);

    if (st->codec->sample_rate <= 0)

        return AVERROR_INVALIDDATA;

    st->codec->channels    = avio_rl32(s->pb);

    if (st->codec->channels <= 0)

        return AVERROR_INVALIDDATA;

    st->duration           = size / (16 * st->codec->channels) * 28;

    align                  = avio_rl32(s->pb);

    if (align <= 0 || align > INT_MAX / st->codec->channels)

        return AVERROR_INVALIDDATA;

    st->codec->block_align = align * st->codec->channels;

    avio_skip(s->pb, 0x800 - avio_tell(s->pb));

    avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);



    return 0;

}
