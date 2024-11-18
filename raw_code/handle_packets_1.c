static int handle_packets(MpegTSContext *ts, int nb_packets)

{

    AVFormatContext *s = ts->stream;

    uint8_t packet[TS_PACKET_SIZE];

    int packet_num, ret = 0;



    if (avio_tell(s->pb) != ts->last_pos) {

        int i;

        av_dlog(ts->stream, "Skipping after seek\n");

        /* seek detected, flush pes buffer */

        for (i = 0; i < NB_PID_MAX; i++) {

            if (ts->pids[i]) {

                if (ts->pids[i]->type == MPEGTS_PES) {

                   PESContext *pes = ts->pids[i]->u.pes_filter.opaque;

                   av_freep(&pes->buffer);

                   pes->data_index = 0;

                   pes->state = MPEGTS_SKIP; /* skip until pes header */

                }

                ts->pids[i]->last_cc = -1;

            }

        }

    }



    ts->stop_parse = 0;

    packet_num = 0;

    for(;;) {

        if (ts->stop_parse>0)

            break;

        packet_num++;

        if (nb_packets != 0 && packet_num >= nb_packets)

            break;

        ret = read_packet(s, packet, ts->raw_packet_size);

        if (ret != 0)

            break;

        ret = handle_packet(ts, packet);

        if (ret != 0)

            break;

    }

    ts->last_pos = avio_tell(s->pb);

    return ret;

}
