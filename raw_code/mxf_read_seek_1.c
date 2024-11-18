static int mxf_read_seek(AVFormatContext *s, int stream_index, int64_t sample_time, int flags)

{

    AVStream *st = s->streams[stream_index];

    int64_t seconds;



    if (!s->bit_rate)

        return AVERROR_INVALIDDATA;

    if (sample_time < 0)

        sample_time = 0;

    seconds = av_rescale(sample_time, st->time_base.num, st->time_base.den);

    avio_seek(s->pb, (s->bit_rate * seconds) >> 3, SEEK_SET);

    ff_update_cur_dts(s, st, sample_time);

    return 0;

}
