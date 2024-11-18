void ff_riff_write_info_tag(AVIOContext *pb, const char *tag, const char *str)

{

    int len = strlen(str);

    if (len > 0) {

        len++;

        ffio_wfourcc(pb, tag);

        avio_wl32(pb, len);

        avio_put_str(pb, str);

        if (len & 1)

            avio_w8(pb, 0);

    }

}
