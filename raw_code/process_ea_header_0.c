static int process_ea_header(AVFormatContext *s) {

    uint32_t blockid, size = 0;

    EaDemuxContext *ea = s->priv_data;

    ByteIOContext *pb = &s->pb;



    blockid = get_le32(pb);

    if (blockid == MVhd_TAG) {

        size = get_le32(pb);

        process_video_header_vp6(s);

        url_fskip(pb, size-32);

        blockid = get_le32(pb);

    }

    if (blockid != SCHl_TAG)

        return 0;

    size += get_le32(pb);

    blockid = get_le32(pb);

    if (blockid == GSTR_TAG) {

        url_fskip(pb, 4);

    } else if (blockid != PT00_TAG) {

        av_log (s, AV_LOG_ERROR, "unknown SCHl headerid\n");

        return 0;

    }



    process_audio_header_elements(s);



    /* skip to the start of the data */

    url_fseek(pb, size, SEEK_SET);



    return 1;

}
