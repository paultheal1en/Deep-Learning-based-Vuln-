int av_read_pause(AVFormatContext *s)

{

    if (s->iformat->read_pause)

        return s->iformat->read_pause(s);

    if (s->pb && s->pb->read_pause)

        return av_url_read_fpause(s->pb, 1);

    return AVERROR(ENOSYS);

}
