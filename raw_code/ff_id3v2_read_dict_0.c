void ff_id3v2_read_dict(AVIOContext *pb, AVDictionary **metadata,

                        const char *magic, ID3v2ExtraMeta **extra_meta)

{

    id3v2_read_internal(pb, metadata, NULL, magic, extra_meta);

}
