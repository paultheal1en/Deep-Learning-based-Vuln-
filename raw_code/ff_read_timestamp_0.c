static int64_t ff_read_timestamp(AVFormatContext *s, int stream_index, int64_t *ppos, int64_t pos_limit,

                                 int64_t (*read_timestamp)(struct AVFormatContext *, int , int64_t *, int64_t ))

{

    return wrap_timestamp(s->streams[stream_index], read_timestamp(s, stream_index, ppos, pos_limit));

}
