static int apc_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    if (av_get_packet(s->pb, pkt, MAX_READ_SIZE) <= 0)

        return AVERROR(EIO);


    pkt->stream_index = 0;

    return 0;

}