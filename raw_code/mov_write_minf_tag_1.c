static int mov_write_minf_tag(AVIOContext *pb, MOVTrack *track)

{

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "minf");

    if (track->enc->codec_type == AVMEDIA_TYPE_VIDEO)

        mov_write_vmhd_tag(pb);

    else if (track->enc->codec_type == AVMEDIA_TYPE_AUDIO)

        mov_write_smhd_tag(pb);

    else if (track->enc->codec_type == AVMEDIA_TYPE_SUBTITLE) {

        if (track->tag == MKTAG('t','e','x','t') || is_clcp_track(track)) {

            mov_write_gmhd_tag(pb, track);

        } else {

            mov_write_nmhd_tag(pb);

        }

    } else if (track->tag == MKTAG('r','t','p',' ')) {

        mov_write_hmhd_tag(pb);

    } else if (track->tag == MKTAG('t','m','c','d')) {

        mov_write_gmhd_tag(pb, track);

    }

    if (track->mode == MODE_MOV) /* FIXME: Why do it for MODE_MOV only ? */

        mov_write_hdlr_tag(pb, NULL);

    mov_write_dinf_tag(pb);

    mov_write_stbl_tag(pb, track);

    return update_size(pb, pos);

}
