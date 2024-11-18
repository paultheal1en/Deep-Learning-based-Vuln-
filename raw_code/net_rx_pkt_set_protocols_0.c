void net_rx_pkt_set_protocols(struct NetRxPkt *pkt, const void *data,

                              size_t len)

{

    assert(pkt);



    eth_get_protocols(data, len, &pkt->isip4, &pkt->isip6,

        &pkt->isudp, &pkt->istcp);

}
