ssize_t qemu_sendv_packet(VLANClientState *sender, const struct iovec *iov,

                          int iovcnt)

{

    VLANState *vlan = sender->vlan;

    VLANClientState *vc;

    VLANPacket *packet;

    ssize_t max_len = 0;

    int i;



    if (sender->link_down)

        return calc_iov_length(iov, iovcnt);



    if (vlan->delivering) {

        max_len = calc_iov_length(iov, iovcnt);



        packet = qemu_malloc(sizeof(VLANPacket) + max_len);

        packet->next = vlan->send_queue;

        packet->sender = sender;

        packet->size = 0;

        for (i = 0; i < iovcnt; i++) {

            size_t len = iov[i].iov_len;



            memcpy(packet->data + packet->size, iov[i].iov_base, len);

            packet->size += len;

        }

        vlan->send_queue = packet;

    } else {

        vlan->delivering = 1;



        for (vc = vlan->first_client; vc != NULL; vc = vc->next) {

            ssize_t len = 0;



            if (vc == sender) {

                continue;

            }

            if (vc->link_down) {

                len = calc_iov_length(iov, iovcnt);

            } else if (vc->receive_iov) {

                len = vc->receive_iov(vc->opaque, iov, iovcnt);

            } else if (vc->receive) {

                len = vc_sendv_compat(vc, iov, iovcnt);

            }

            max_len = MAX(max_len, len);

        }



        while ((packet = vlan->send_queue) != NULL) {

            vlan->send_queue = packet->next;

            qemu_deliver_packet(packet->sender, packet->data, packet->size);

            qemu_free(packet);

        }

        vlan->delivering = 0;

    }



    return max_len;

}
