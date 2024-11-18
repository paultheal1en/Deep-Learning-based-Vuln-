static int mov_write_stbl_tag(AVFormatContext *s, AVIOContext *pb, MOVMuxContext *mov, MOVTrack *track)

{

    int64_t pos = avio_tell(pb);

    int ret;



    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "stbl");

    mov_write_stsd_tag(s, pb, mov, track);

    mov_write_stts_tag(pb, track);

    if ((track->par->codec_type == AVMEDIA_TYPE_VIDEO ||

         track->par->codec_tag == MKTAG('r','t','p',' ')) &&

        track->has_keyframes && track->has_keyframes < track->entry)

        mov_write_stss_tag(pb, track, MOV_SYNC_SAMPLE);

    if (track->mode == MODE_MOV && track->flags & MOV_TRACK_STPS)

        mov_write_stss_tag(pb, track, MOV_PARTIAL_SYNC_SAMPLE);

    if (track->par->codec_type == AVMEDIA_TYPE_VIDEO &&

        track->flags & MOV_TRACK_CTTS && track->entry) {



        if ((ret = mov_write_ctts_tag(pb, track)) < 0)

            return ret;

    }

    mov_write_stsc_tag(pb, track);

    mov_write_stsz_tag(pb, track);

    mov_write_stco_tag(pb, track);

    if (mov->encryption_scheme == MOV_ENC_CENC_AES_CTR) {

        ff_mov_cenc_write_stbl_atoms(&track->cenc, pb);

    }

    if (track->par->codec_id == AV_CODEC_ID_OPUS) {

        mov_preroll_write_stbl_atoms(pb, track);

    }

    return update_size(pb, pos);

}
