int64_t ff_ape_parse_tag(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    int file_size = avio_size(pb);

    uint32_t val, fields, tag_bytes;

    uint8_t buf[8];

    int64_t tag_start;

    int i;



    if (file_size < APE_TAG_FOOTER_BYTES)

        return 0;



    avio_seek(pb, file_size - APE_TAG_FOOTER_BYTES, SEEK_SET);



    avio_read(pb, buf, 8);     /* APETAGEX */

    if (strncmp(buf, "APETAGEX", 8)) {

        return 0;

    }



    val = avio_rl32(pb);       /* APE tag version */

    if (val > APE_TAG_VERSION) {

        av_log(s, AV_LOG_ERROR, "Unsupported tag version. (>=%d)\n", APE_TAG_VERSION);

        return 0;

    }



    tag_bytes = avio_rl32(pb); /* tag size */

    if (tag_bytes - APE_TAG_FOOTER_BYTES > (1024 * 1024 * 16)) {

        av_log(s, AV_LOG_ERROR, "Tag size is way too big\n");

        return 0;

    }



    tag_start = file_size - tag_bytes - APE_TAG_FOOTER_BYTES;

    if (tag_start < 0) {

        av_log(s, AV_LOG_ERROR, "Invalid tag size %u.\n", tag_bytes);

        return 0;

    }



    fields = avio_rl32(pb);    /* number of fields */

    if (fields > 65536) {

        av_log(s, AV_LOG_ERROR, "Too many tag fields (%d)\n", fields);

        return 0;

    }



    val = avio_rl32(pb);       /* flags */

    if (val & APE_TAG_FLAG_IS_HEADER) {

        av_log(s, AV_LOG_ERROR, "APE Tag is a header\n");

        return 0;

    }



    avio_seek(pb, file_size - tag_bytes, SEEK_SET);



    for (i=0; i<fields; i++)

        if (ape_tag_read_field(s) < 0) break;



    return tag_start;

}
