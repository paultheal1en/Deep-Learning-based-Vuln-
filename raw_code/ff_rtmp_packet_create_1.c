int ff_rtmp_packet_create(RTMPPacket *pkt, int channel_id, RTMPPacketType type,

                          int timestamp, int size)

{


    pkt->data = av_malloc(size);

    if (!pkt->data)

        return AVERROR(ENOMEM);


    pkt->data_size  = size;

    pkt->channel_id = channel_id;

    pkt->type       = type;

    pkt->timestamp  = timestamp;

    pkt->extra      = 0;

    pkt->ts_delta   = 0;



    return 0;
