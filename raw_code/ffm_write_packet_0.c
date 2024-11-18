static int ffm_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    FFMContext *ffm = s->priv_data;

    AVStream *st = s->streams[pkt->stream_index];

    int64_t pts;

    uint8_t header[FRAME_HEADER_SIZE];



    pts = ffm->start_time + pkt->pts;

    /* packet size & key_frame */

    header[0] = pkt->stream_index;

    header[1] = 0;

    if (pkt->flags & PKT_FLAG_KEY)

        header[1] |= FLAG_KEY_FRAME;

    AV_WB24(header+2, pkt->size);

    AV_WB24(header+5, pkt->duration);

    ffm_write_data(s, header, FRAME_HEADER_SIZE, pts, 1);

    ffm_write_data(s, pkt->data, pkt->size, pts, 0);



    return 0;

}
