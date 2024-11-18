static int mpegts_read_packet(AVFormatContext *s,

                              AVPacket *pkt)

{

    MpegTSContext *ts = s->priv_data;



    if (!ts->mpeg2ts_raw) {

        ts->pkt = pkt;

        return handle_packets(ts, 0);

    } else {

        return mpegts_raw_read_packet(s, pkt);

    }

}
