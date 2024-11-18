static int fits_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    write_image_header(s);

    avio_write(s->pb, pkt->data, pkt->size);

    return 0;

}
