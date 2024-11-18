static void read_id3(AVFormatContext *s, uint64_t id3pos)

{

    ID3v2ExtraMeta *id3v2_extra_meta = NULL;

    if (avio_seek(s->pb, id3pos, SEEK_SET) < 0)

        return;



    ff_id3v2_read(s, ID3v2_DEFAULT_MAGIC, &id3v2_extra_meta);

    if (id3v2_extra_meta)

        ff_id3v2_parse_apic(s, &id3v2_extra_meta);

    ff_id3v2_free_extra_meta(&id3v2_extra_meta);

}
