e1000e_rss_get_hash_type(E1000ECore *core, struct NetRxPkt *pkt)

{

    bool isip4, isip6, isudp, istcp;



    assert(e1000e_rss_enabled(core));



    net_rx_pkt_get_protocols(pkt, &isip4, &isip6, &isudp, &istcp);



    if (isip4) {

        bool fragment = net_rx_pkt_get_ip4_info(pkt)->fragment;



        trace_e1000e_rx_rss_ip4(fragment, istcp, core->mac[MRQC],

                                E1000_MRQC_EN_TCPIPV4(core->mac[MRQC]),

                                E1000_MRQC_EN_IPV4(core->mac[MRQC]));



        if (!fragment && istcp && E1000_MRQC_EN_TCPIPV4(core->mac[MRQC])) {

            return E1000_MRQ_RSS_TYPE_IPV4TCP;

        }



        if (E1000_MRQC_EN_IPV4(core->mac[MRQC])) {

            return E1000_MRQ_RSS_TYPE_IPV4;

        }

    } else if (isip6) {

        eth_ip6_hdr_info *ip6info = net_rx_pkt_get_ip6_info(pkt);



        bool ex_dis = core->mac[RFCTL] & E1000_RFCTL_IPV6_EX_DIS;

        bool new_ex_dis = core->mac[RFCTL] & E1000_RFCTL_NEW_IPV6_EXT_DIS;



        trace_e1000e_rx_rss_ip6(core->mac[RFCTL],

                                ex_dis, new_ex_dis, istcp,

                                ip6info->has_ext_hdrs,

                                ip6info->rss_ex_dst_valid,

                                ip6info->rss_ex_src_valid,

                                core->mac[MRQC],

                                E1000_MRQC_EN_TCPIPV6(core->mac[MRQC]),

                                E1000_MRQC_EN_IPV6EX(core->mac[MRQC]),

                                E1000_MRQC_EN_IPV6(core->mac[MRQC]));



        if ((!ex_dis || !ip6info->has_ext_hdrs) &&

            (!new_ex_dis || !(ip6info->rss_ex_dst_valid ||

                              ip6info->rss_ex_src_valid))) {



            if (istcp && !ip6info->fragment &&

                E1000_MRQC_EN_TCPIPV6(core->mac[MRQC])) {

                return E1000_MRQ_RSS_TYPE_IPV6TCP;

            }



            if (E1000_MRQC_EN_IPV6EX(core->mac[MRQC])) {

                return E1000_MRQ_RSS_TYPE_IPV6EX;

            }



        }



        if (E1000_MRQC_EN_IPV6(core->mac[MRQC])) {

            return E1000_MRQ_RSS_TYPE_IPV6;

        }



    }



    return E1000_MRQ_RSS_TYPE_NONE;

}
