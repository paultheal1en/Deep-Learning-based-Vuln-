static int asf_read_seek(AVFormatContext *s, int stream_index,

                         int64_t pts, int flags)

{

    ASFContext *asf = s->priv_data;

    AVStream *st    = s->streams[stream_index];

    int64_t pos;

    int index;



    if (s->packet_size <= 0)

        return -1;



    /* Try using the protocol's read_seek if available */

    if (s->pb) {

        int ret = avio_seek_time(s->pb, stream_index, pts, flags);

        if (ret >= 0)

            asf_reset_header(s);

        if (ret != AVERROR(ENOSYS))

            return ret;

    }



    if (!asf->index_read)

        asf_build_simple_index(s, stream_index);



    if ((asf->index_read && st->index_entries)) {

        index = av_index_search_timestamp(st, pts, flags);

        if (index >= 0) {

            /* find the position */

            pos = st->index_entries[index].pos;



            /* do the seek */

            av_log(s, AV_LOG_DEBUG, "SEEKTO: %"PRId64"\n", pos);

            avio_seek(s->pb, pos, SEEK_SET);

            asf_reset_header(s);

            return 0;

        }

    }

    /* no index or seeking by index failed */

    if (ff_seek_frame_binary(s, stream_index, pts, flags) < 0)

        return -1;

    asf_reset_header(s);

    return 0;

}
