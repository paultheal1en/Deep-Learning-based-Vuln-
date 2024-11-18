static int tta_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    TTAContext *c = s->priv_data;

    AVStream *st = s->streams[0];

    int size, ret;



    // FIXME!

    if (c->currentframe > c->totalframes)

        return -1;



    size = st->index_entries[c->currentframe].size;



    ret = av_get_packet(s->pb, pkt, size);

    pkt->dts = st->index_entries[c->currentframe++].timestamp;

    return ret;

}
