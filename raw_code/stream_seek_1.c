static void stream_seek(VideoState *is, int64_t pos, int rel)

{

    is->seek_pos = pos;

    is->seek_req = 1;

    is->seek_flags = rel < 0 ? AVSEEK_FLAG_BACKWARD : 0;

}
