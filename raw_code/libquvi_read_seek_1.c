static int libquvi_read_seek(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    LibQuviContext *qc = s->priv_data;

    return av_seek_frame(qc->fmtctx, stream_index, timestamp, flags);

}
