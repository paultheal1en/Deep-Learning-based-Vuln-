static int64_t rtmp_read_seek(URLContext *s, int stream_index,

                              int64_t timestamp, int flags)

{

    RTMP *r = s->priv_data;



    if (flags & AVSEEK_FLAG_BYTE)

        return AVERROR(ENOSYS);



    /* seeks are in milliseconds */

    timestamp = av_rescale(timestamp, AV_TIME_BASE, 1000);

    if (!RTMP_SendSeek(r, timestamp))

        return -1;

    return timestamp;

}
