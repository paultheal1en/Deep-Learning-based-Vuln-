static int handle_packets(AVFormatContext *s, int nb_packets)

{

    MpegTSContext *ts = s->priv_data;

    ByteIOContext *pb = &s->pb;

    uint8_t packet[TS_FEC_PACKET_SIZE];

    int packet_num, len;



    ts->stop_parse = 0;

    packet_num = 0;

    for(;;) {

        if (ts->stop_parse)

            break;

        packet_num++;

        if (nb_packets != 0 && packet_num >= nb_packets)

            break;

        len = get_buffer(pb, packet, ts->raw_packet_size);

        if (len != ts->raw_packet_size)

            return AVERROR_IO;

        /* check paquet sync byte */

        /* XXX: accept to resync ? */

        if (packet[0] != 0x47)

            return AVERROR_INVALIDDATA;

        handle_packet(s, packet);

    }

    return 0;

}
