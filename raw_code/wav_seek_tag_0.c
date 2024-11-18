static int64_t wav_seek_tag(AVIOContext *s, int64_t offset, int whence)

{

    offset += offset < INT64_MAX && offset & 1;



    return avio_seek(s, offset, whence);

}
