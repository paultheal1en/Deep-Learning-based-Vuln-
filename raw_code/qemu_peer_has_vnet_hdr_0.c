bool qemu_peer_has_vnet_hdr(NetClientState *nc)

{

    if (!nc->peer || !nc->peer->info->has_vnet_hdr) {

        return false;

    }



    return nc->peer->info->has_vnet_hdr(nc->peer);

}
