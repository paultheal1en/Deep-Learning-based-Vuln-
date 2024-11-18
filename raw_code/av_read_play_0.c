int av_read_play(AVFormatContext *s)

{

    if (s->iformat->read_play)

        return s->iformat->read_play(s);

    if (s->pb && s->pb->read_pause)

        return av_url_read_fpause(s->pb, 0);

    return AVERROR(ENOSYS);

}
