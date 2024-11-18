void qemu_peer_set_vnet_hdr_len(NetClientState *nc, int len)

{

    if (!nc->peer || !nc->peer->info->set_vnet_hdr_len) {

        return;

    }



    nc->peer->info->set_vnet_hdr_len(nc->peer, len);

}
