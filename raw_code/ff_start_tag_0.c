int64_t ff_start_tag(AVIOContext *pb, const char *tag)

{

    ffio_wfourcc(pb, tag);

    avio_wl32(pb, 0);

    return avio_tell(pb);

}
