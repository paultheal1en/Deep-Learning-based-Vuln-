void ff_id3v2_read(AVFormatContext *s, const char *magic,

                   ID3v2ExtraMeta **extra_meta)

{

    id3v2_read_internal(s->pb, &s->metadata, s, magic, extra_meta);

}
