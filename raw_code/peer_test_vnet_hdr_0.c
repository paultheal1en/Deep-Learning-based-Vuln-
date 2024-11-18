static void peer_test_vnet_hdr(VirtIONet *n)

{

    NetClientState *nc = qemu_get_queue(n->nic);

    if (!nc->peer) {

        return;

    }



    n->has_vnet_hdr = qemu_peer_has_vnet_hdr(nc);

}
