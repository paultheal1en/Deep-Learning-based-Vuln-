static int mov_write_hdlr_tag(AVIOContext *pb, MOVTrack *track)

{

    const char *hdlr, *descr = NULL, *hdlr_type = NULL;

    int64_t pos = avio_tell(pb);



    if (!track) { /* no media --> data handler */





        hdlr = (track->mode == MODE_MOV) ? "mhlr" : "\0\0\0\0";

        if (track->enc->codec_type == AVMEDIA_TYPE_VIDEO) {

            hdlr_type = "vide";

            descr = "VideoHandler";

        } else if (track->enc->codec_type == AVMEDIA_TYPE_AUDIO) {

            hdlr_type = "soun";

            descr = "SoundHandler";

        } else if (track->enc->codec_type == AVMEDIA_TYPE_SUBTITLE) {

            if (track->tag == MKTAG('t','x','3','g')) hdlr_type = "sbtl";

            else                                      hdlr_type = "text";

            descr = "SubtitleHandler";

        } else if (track->enc->codec_tag == MKTAG('r','t','p',' ')) {

            hdlr_type = "hint";

            descr = "HintHandler";





        }

    }



    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "hdlr");

    avio_wb32(pb, 0); /* Version & flags */

    avio_write(pb, hdlr, 4); /* handler */

    ffio_wfourcc(pb, hdlr_type); /* handler type */

    avio_wb32(pb ,0); /* reserved */

    avio_wb32(pb ,0); /* reserved */

    avio_wb32(pb ,0); /* reserved */

    if (!track || track->mode == MODE_MOV)

        avio_w8(pb, strlen(descr)); /* pascal string */

    avio_write(pb, descr, strlen(descr)); /* handler description */

    if (track && track->mode != MODE_MOV)

        avio_w8(pb, 0); /* c string */

    return update_size(pb, pos);

}