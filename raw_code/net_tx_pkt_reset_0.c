void net_tx_pkt_reset(struct NetTxPkt *pkt)

{

    int i;



    /* no assert, as reset can be called before tx_pkt_init */

    if (!pkt) {

        return;

    }



    memset(&pkt->virt_hdr, 0, sizeof(pkt->virt_hdr));



    g_free(pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_base);

    pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_base = NULL;



    assert(pkt->vec);

    for (i = NET_TX_PKT_L2HDR_FRAG;

         i < pkt->payload_frags + NET_TX_PKT_PL_START_FRAG; i++) {

        pkt->vec[i].iov_len = 0;

    }

    pkt->payload_len = 0;

    pkt->payload_frags = 0;



    assert(pkt->raw);

    for (i = 0; i < pkt->raw_frags; i++) {

        assert(pkt->raw[i].iov_base);

        cpu_physical_memory_unmap(pkt->raw[i].iov_base, pkt->raw[i].iov_len,

                                  false, pkt->raw[i].iov_len);

        pkt->raw[i].iov_len = 0;

    }

    pkt->raw_frags = 0;



    pkt->hdr_len = 0;

    pkt->packet_type = 0;

    pkt->l4proto = 0;

}
