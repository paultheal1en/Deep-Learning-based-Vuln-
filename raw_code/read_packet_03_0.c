static int read_packet(AVFormatContext *s, AVPacket *pkt)

{

    ASSContext *ass = s->priv_data;

    uint8_t *p, *end;



    if (ass->event_index >= ass->event_count)

        return AVERROR(EIO);



    p = ass->event[ass->event_index];



    end = strchr(p, '\n');

    av_new_packet(pkt, end ? end - p + 1 : strlen(p));

    pkt->flags |= AV_PKT_FLAG_KEY;

    pkt->pos    = p - ass->event_buffer + s->streams[0]->codec->extradata_size;

    pkt->pts    = pkt->dts = get_pts(p);

    memcpy(pkt->data, p, pkt->size);



    ass->event_index++;



    return 0;

}
