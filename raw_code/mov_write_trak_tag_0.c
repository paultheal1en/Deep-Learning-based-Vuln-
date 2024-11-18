static int mov_write_trak_tag(AVIOContext *pb, MOVMuxContext *mov,

                              MOVTrack *track, AVStream *st)

{

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "trak");

    mov_write_tkhd_tag(pb, track, st);

    if (supports_edts(mov))

        mov_write_edts_tag(pb, track);  // PSP Movies and several other cases require edts box

    if (track->tref_tag)

        mov_write_tref_tag(pb, track);

    mov_write_mdia_tag(pb, track);

    if (track->mode == MODE_PSP)

        mov_write_uuid_tag_psp(pb, track); // PSP Movies require this uuid box

    if (track->tag == MKTAG('r','t','p',' '))

        mov_write_udta_sdp(pb, track);

    if (track->mode == MODE_MOV) {

        if (track->enc->codec_type == AVMEDIA_TYPE_VIDEO) {

            double sample_aspect_ratio = av_q2d(st->sample_aspect_ratio);

            if (st->sample_aspect_ratio.num && 1.0 != sample_aspect_ratio) {

                mov_write_tapt_tag(pb, track);

            }

        }

        if (is_clcp_track(track)) {

            mov_write_tapt_tag(pb, track);

        }

    }

    return update_size(pb, pos);

}
