void eth_get_protocols(const struct iovec *iov, int iovcnt,

                       bool *isip4, bool *isip6,

                       bool *isudp, bool *istcp,

                       size_t *l3hdr_off,

                       size_t *l4hdr_off,

                       size_t *l5hdr_off,

                       eth_ip6_hdr_info *ip6hdr_info,

                       eth_ip4_hdr_info *ip4hdr_info,

                       eth_l4_hdr_info  *l4hdr_info)

{

    int proto;

    bool fragment = false;

    size_t l2hdr_len = eth_get_l2_hdr_length_iov(iov, iovcnt);

    size_t input_size = iov_size(iov, iovcnt);

    size_t copied;



    *isip4 = *isip6 = *isudp = *istcp = false;



    proto = eth_get_l3_proto(iov, iovcnt, l2hdr_len);



    *l3hdr_off = l2hdr_len;



    if (proto == ETH_P_IP) {

        struct ip_header *iphdr = &ip4hdr_info->ip4_hdr;



        if (input_size < l2hdr_len) {

            return;

        }



        copied = iov_to_buf(iov, iovcnt, l2hdr_len, iphdr, sizeof(*iphdr));



        *isip4 = true;



        if (copied < sizeof(*iphdr)) {

            return;

        }



        if (IP_HEADER_VERSION(iphdr) == IP_HEADER_VERSION_4) {

            if (iphdr->ip_p == IP_PROTO_TCP) {

                *istcp = true;

            } else if (iphdr->ip_p == IP_PROTO_UDP) {

                *isudp = true;

            }

        }



        ip4hdr_info->fragment = IP4_IS_FRAGMENT(iphdr);

        *l4hdr_off = l2hdr_len + IP_HDR_GET_LEN(iphdr);



        fragment = ip4hdr_info->fragment;

    } else if (proto == ETH_P_IPV6) {



        *isip6 = true;

        if (eth_parse_ipv6_hdr(iov, iovcnt, l2hdr_len,

                               ip6hdr_info)) {

            if (ip6hdr_info->l4proto == IP_PROTO_TCP) {

                *istcp = true;

            } else if (ip6hdr_info->l4proto == IP_PROTO_UDP) {

                *isudp = true;

            }

        } else {

            return;

        }



        *l4hdr_off = l2hdr_len + ip6hdr_info->full_hdr_len;

        fragment = ip6hdr_info->fragment;

    }



    if (!fragment) {

        if (*istcp) {

            *istcp = _eth_copy_chunk(input_size,

                                     iov, iovcnt,

                                     *l4hdr_off, sizeof(l4hdr_info->hdr.tcp),

                                     &l4hdr_info->hdr.tcp);



            if (istcp) {

                *l5hdr_off = *l4hdr_off +

                    TCP_HEADER_DATA_OFFSET(&l4hdr_info->hdr.tcp);



                l4hdr_info->has_tcp_data =

                    _eth_tcp_has_data(proto == ETH_P_IP,

                                      &ip4hdr_info->ip4_hdr,

                                      &ip6hdr_info->ip6_hdr,

                                      *l4hdr_off - *l3hdr_off,

                                      &l4hdr_info->hdr.tcp);

            }

        } else if (*isudp) {

            *isudp = _eth_copy_chunk(input_size,

                                     iov, iovcnt,

                                     *l4hdr_off, sizeof(l4hdr_info->hdr.udp),

                                     &l4hdr_info->hdr.udp);

            *l5hdr_off = *l4hdr_off + sizeof(l4hdr_info->hdr.udp);

        }

    }

}
