void net_tx_pkt_setup_vlan_header(struct NetTxPkt *pkt, uint16_t vlan)

{

    bool is_new;

    assert(pkt);



    eth_setup_vlan_headers(pkt->vec[NET_TX_PKT_L2HDR_FRAG].iov_base,

        vlan, &is_new);



    /* update l2hdrlen */

    if (is_new) {

        pkt->hdr_len += sizeof(struct vlan_header);

        pkt->vec[NET_TX_PKT_L2HDR_FRAG].iov_len +=

            sizeof(struct vlan_header);

    }

}
