qemu_deliver_packet(VLANClientState *sender, const uint8_t *buf, int size)

{

    VLANClientState *vc;



    for (vc = sender->vlan->first_client; vc != NULL; vc = vc->next) {

        if (vc != sender && !vc->link_down) {

            vc->receive(vc->opaque, buf, size);

        }

    }

}
