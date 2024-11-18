int av_append_packet(AVIOContext *s, AVPacket *pkt, int size)

{

    int ret;

    int old_size;

    if (!pkt->size)

        return av_get_packet(s, pkt, size);

    old_size = pkt->size;

    ret = av_grow_packet(pkt, size);

    if (ret < 0)

        return ret;

    ret = avio_read(s, pkt->data + old_size, size);

    av_shrink_packet(pkt, old_size + FFMAX(ret, 0));

    return ret;

}
