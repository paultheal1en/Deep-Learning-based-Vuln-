static int64_t ogg_read_timestamp(AVFormatContext *s, int stream_index,

                                  int64_t *pos_arg, int64_t pos_limit)

{

    struct ogg *ogg = s->priv_data;

    struct ogg_stream *os = ogg->streams + stream_index;

    AVIOContext *bc = s->pb;

    int64_t pts = AV_NOPTS_VALUE;

    int i;

    avio_seek(bc, *pos_arg, SEEK_SET);

    ogg_reset(ogg);



    while (avio_tell(bc) < pos_limit && !ogg_packet(s, &i, NULL, NULL, pos_arg)) {

        if (i == stream_index) {

            pts = ogg_calc_pts(s, i, NULL);

            if (os->keyframe_seek && !(os->pflags & AV_PKT_FLAG_KEY))

                pts = AV_NOPTS_VALUE;

        }

        if (pts != AV_NOPTS_VALUE)

            break;

    }

    ogg_reset(ogg);

    return pts;

}
