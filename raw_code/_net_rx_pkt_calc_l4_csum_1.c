_net_rx_pkt_calc_l4_csum(struct NetRxPkt *pkt)

{

    uint32_t cntr;

    uint16_t csum;

    uint16_t csl;

    uint32_t cso;



    trace_net_rx_pkt_l4_csum_calc_entry();



    if (pkt->isip4) {

        if (pkt->isudp) {

            csl = be16_to_cpu(pkt->l4hdr_info.hdr.udp.uh_ulen);

            trace_net_rx_pkt_l4_csum_calc_ip4_udp();

        } else {

            csl = be16_to_cpu(pkt->ip4hdr_info.ip4_hdr.ip_len) -

                  IP_HDR_GET_LEN(&pkt->ip4hdr_info.ip4_hdr);

            trace_net_rx_pkt_l4_csum_calc_ip4_tcp();

        }



        cntr = eth_calc_ip4_pseudo_hdr_csum(&pkt->ip4hdr_info.ip4_hdr,

                                            csl, &cso);

        trace_net_rx_pkt_l4_csum_calc_ph_csum(cntr, csl);

    } else {

        if (pkt->isudp) {

            csl = be16_to_cpu(pkt->l4hdr_info.hdr.udp.uh_ulen);

            trace_net_rx_pkt_l4_csum_calc_ip6_udp();

        } else {

            struct ip6_header *ip6hdr = &pkt->ip6hdr_info.ip6_hdr;

            size_t full_ip6hdr_len = pkt->l4hdr_off - pkt->l3hdr_off;

            size_t ip6opts_len = full_ip6hdr_len - sizeof(struct ip6_header);



            csl = be16_to_cpu(ip6hdr->ip6_ctlun.ip6_un1.ip6_un1_plen) -

                  ip6opts_len;

            trace_net_rx_pkt_l4_csum_calc_ip6_tcp();

        }



        cntr = eth_calc_ip6_pseudo_hdr_csum(&pkt->ip6hdr_info.ip6_hdr, csl,

                                            pkt->ip6hdr_info.l4proto, &cso);

        trace_net_rx_pkt_l4_csum_calc_ph_csum(cntr, csl);

    }



    cntr += net_checksum_add_iov(pkt->vec, pkt->vec_len,

                                 pkt->l4hdr_off, csl, cso);



    csum = net_checksum_finish(cntr);



    trace_net_rx_pkt_l4_csum_calc_csum(pkt->l4hdr_off, csl, cntr, csum);



    return csum;

}
