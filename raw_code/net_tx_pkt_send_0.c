bool net_tx_pkt_send(struct NetTxPkt *pkt, NetClientState *nc)

{

    assert(pkt);



    if (!pkt->has_virt_hdr &&

        pkt->virt_hdr.flags & VIRTIO_NET_HDR_F_NEEDS_CSUM) {

        net_tx_pkt_do_sw_csum(pkt);

    }



    /*

     * Since underlying infrastructure does not support IP datagrams longer

     * than 64K we should drop such packets and don't even try to send

     */

    if (VIRTIO_NET_HDR_GSO_NONE != pkt->virt_hdr.gso_type) {

        if (pkt->payload_len >

            ETH_MAX_IP_DGRAM_LEN -

            pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_len) {

            return false;

        }

    }



    if (pkt->has_virt_hdr ||

        pkt->virt_hdr.gso_type == VIRTIO_NET_HDR_GSO_NONE) {

        qemu_sendv_packet(nc, pkt->vec,

            pkt->payload_frags + NET_TX_PKT_PL_START_FRAG);

        return true;

    }



    return net_tx_pkt_do_sw_fragmentation(pkt, nc);

}
