static int av_seek_frame_generic(AVFormatContext *s, 

                                 int stream_index, int64_t timestamp)

{

    int index;

    AVStream *st;

    AVIndexEntry *ie;



    if (!s->index_built) {

        if (is_raw_stream(s)) {

            av_build_index_raw(s);

        } else {

            return -1;

        }

        s->index_built = 1;

    }



    if (stream_index < 0)

        stream_index = 0;

    st = s->streams[stream_index];

    index = index_search_timestamp(st->index_entries, st->nb_index_entries,

                                   timestamp);

    if (index < 0)

        return -1;



    /* now we have found the index, we can seek */

    ie = &st->index_entries[index];

    av_read_frame_flush(s);

    url_fseek(&s->pb, ie->pos, SEEK_SET);

    st->cur_dts = ie->timestamp;

    return 0;

}
