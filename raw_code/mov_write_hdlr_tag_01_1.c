static int mov_write_hdlr_tag(ByteIOContext *pb, MOVTrack *track)

{

    const char *descr, *hdlr, *hdlr_type;

    int64_t pos = url_ftell(pb);



    if (!track) { /* no media --> data handler */

        hdlr = "dhlr";

        hdlr_type = "url ";

        descr = "DataHandler";

    } else {

        hdlr = (track->mode == MODE_MOV) ? "mhlr" : "\0\0\0\0";

        if (track->enc->codec_type == CODEC_TYPE_VIDEO) {

            hdlr_type = "vide";

            descr = "VideoHandler";

        } else if (track->enc->codec_type == CODEC_TYPE_AUDIO){

            hdlr_type = "soun";

            descr = "SoundHandler";

        } else if (track->enc->codec_type == CODEC_TYPE_SUBTITLE){

            if (track->mode == MODE_IPOD) hdlr_type = "sbtl";

            else                          hdlr_type = "text";

            descr = "SubtitleHandler";

        }

    }



    put_be32(pb, 0); /* size */

    put_tag(pb, "hdlr");

    put_be32(pb, 0); /* Version & flags */

    put_buffer(pb, hdlr, 4); /* handler */

    put_tag(pb, hdlr_type); /* handler type */

    put_be32(pb ,0); /* reserved */

    put_be32(pb ,0); /* reserved */

    put_be32(pb ,0); /* reserved */

    put_byte(pb, strlen(descr)); /* string counter */

    put_buffer(pb, descr, strlen(descr)); /* handler description */

    return updateSize(pb, pos);

}
