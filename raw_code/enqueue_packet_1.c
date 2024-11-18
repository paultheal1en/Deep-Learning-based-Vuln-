static void enqueue_packet(RTPDemuxContext *s, uint8_t *buf, int len)

{

    uint16_t seq   = AV_RB16(buf + 2);

    RTPPacket **cur = &s->queue, *packet;



    /* Find the correct place in the queue to insert the packet */

    while (*cur) {

        int16_t diff = seq - (*cur)->seq;

        if (diff < 0)

            break;

        cur = &(*cur)->next;

    }



    packet = av_mallocz(sizeof(*packet));

    if (!packet)

        return;

    packet->recvtime = av_gettime_relative();

    packet->seq      = seq;

    packet->len      = len;

    packet->buf      = buf;

    packet->next     = *cur;

    *cur = packet;

    s->queue_len++;

}
