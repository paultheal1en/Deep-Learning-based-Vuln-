static int iff_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    IffDemuxContext *iff = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st = s->streams[0];

    int ret;

    int64_t pos = avio_tell(pb);



    if (pos >= iff->body_end)

        return AVERROR_EOF;



    if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

        if (st->codec->codec_tag == ID_MAUD) {

            ret = av_get_packet(pb, pkt, FFMIN(iff->body_end - pos, 1024 * st->codec->block_align));

        } else {

            ret = av_get_packet(pb, pkt, iff->body_size);

        }

    } else if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

        uint8_t *buf;



        if (av_new_packet(pkt, iff->body_size + 2) < 0) {

            return AVERROR(ENOMEM);

        }



        buf = pkt->data;

        bytestream_put_be16(&buf, 2);

        ret = avio_read(pb, buf, iff->body_size);



    } else {

        av_assert0(0);

    }



    if (pos == iff->body_pos)

        pkt->flags |= AV_PKT_FLAG_KEY;

    if (ret < 0)

        return ret;

    pkt->stream_index = 0;

    return ret;

}