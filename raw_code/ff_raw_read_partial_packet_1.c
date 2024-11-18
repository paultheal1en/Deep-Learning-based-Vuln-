int ff_raw_read_partial_packet(AVFormatContext *s, AVPacket *pkt)

{

    int ret, size;



    size = RAW_PACKET_SIZE;



    if (av_new_packet(pkt, size) < 0)

        return AVERROR(ENOMEM);



    pkt->pos= avio_tell(s->pb);

    pkt->stream_index = 0;

    ret = ffio_read_partial(s->pb, pkt->data, size);

    if (ret < 0) {

        av_free_packet(pkt);

        return ret;

    }

    pkt->size = ret;

    return ret;

}
