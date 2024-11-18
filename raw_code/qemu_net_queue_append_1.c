static void qemu_net_queue_append(NetQueue *queue,

                                  NetClientState *sender,

                                  unsigned flags,

                                  const uint8_t *buf,

                                  size_t size,

                                  NetPacketSent *sent_cb)

{

    NetPacket *packet;



    if (queue->nq_count >= queue->nq_maxlen && !sent_cb) {

        return; /* drop if queue full and no callback */

    }

    packet = g_malloc(sizeof(NetPacket) + size);

    packet->sender = sender;

    packet->flags = flags;

    packet->size = size;

    packet->sent_cb = sent_cb;

    memcpy(packet->data, buf, size);




    QTAILQ_INSERT_TAIL(&queue->packets, packet, entry);

}