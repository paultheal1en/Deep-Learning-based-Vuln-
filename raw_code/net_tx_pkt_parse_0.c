bool net_tx_pkt_parse(struct NetTxPkt *pkt)

{

    return net_tx_pkt_parse_headers(pkt) &&

           net_tx_pkt_rebuild_payload(pkt);

}
