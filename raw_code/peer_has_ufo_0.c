static int peer_has_ufo(VirtIONet *n)

{

    if (!peer_has_vnet_hdr(n))

        return 0;



    n->has_ufo = qemu_peer_has_ufo(qemu_get_queue(n->nic));



    return n->has_ufo;

}
