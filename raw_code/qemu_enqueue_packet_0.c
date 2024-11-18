static void qemu_enqueue_packet(VLANClientState *sender,

                                const uint8_t *buf, int size,

                                NetPacketSent *sent_cb)

{

    VLANPacket *packet;



    packet = qemu_malloc(sizeof(VLANPacket) + size);

    packet->sender = sender;

    packet->size = size;

    packet->sent_cb = sent_cb;

    memcpy(packet->data, buf, size);



    TAILQ_INSERT_TAIL(&sender->vlan->send_queue, packet, entry);

}
