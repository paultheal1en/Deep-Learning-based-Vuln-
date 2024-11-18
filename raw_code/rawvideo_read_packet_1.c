static int rawvideo_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int packet_size, ret, width, height;

    AVStream *st = s->streams[0];



    width = st->codec->width;

    height = st->codec->height;



    packet_size = avpicture_get_size(st->codec->pix_fmt, width, height);

    if (packet_size < 0)

        return -1;



    ret= av_get_packet(s->pb, pkt, packet_size);

    pkt->pts=

    pkt->dts= pkt->pos / packet_size;



    pkt->stream_index = 0;

    if (ret != packet_size)

        return AVERROR(EIO);

    return 0;

}
