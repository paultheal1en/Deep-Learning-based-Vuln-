void qemu_purge_queued_packets(VLANClientState *vc)

{

    VLANPacket *packet, *next;



    TAILQ_FOREACH_SAFE(packet, &vc->vlan->send_queue, entry, next) {

        if (packet->sender == vc) {

            TAILQ_REMOVE(&vc->vlan->send_queue, packet, entry);

            qemu_free(packet);

        }

    }

}
