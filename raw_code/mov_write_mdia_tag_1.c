static int mov_write_mdia_tag(AVIOContext *pb, MOVMuxContext *mov,

                              MOVTrack *track)

{

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "mdia");

    mov_write_mdhd_tag(pb, mov, track);

    mov_write_hdlr_tag(pb, track);

    mov_write_minf_tag(pb, track);

    return update_size(pb, pos);

}
