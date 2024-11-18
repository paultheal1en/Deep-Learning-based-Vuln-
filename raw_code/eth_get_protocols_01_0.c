void eth_get_protocols(const uint8_t *headers,

                       uint32_t hdr_length,

                       bool *isip4, bool *isip6,

                       bool *isudp, bool *istcp)

{

    int proto;

    size_t l2hdr_len = eth_get_l2_hdr_length(headers);

    assert(hdr_length >= eth_get_l2_hdr_length(headers));

    *isip4 = *isip6 = *isudp = *istcp = false;



    proto = eth_get_l3_proto(headers, l2hdr_len);

    if (proto == ETH_P_IP) {

        *isip4 = true;



        struct ip_header *iphdr;



        assert(hdr_length >=

            eth_get_l2_hdr_length(headers) + sizeof(struct ip_header));



        iphdr = PKT_GET_IP_HDR(headers);



        if (IP_HEADER_VERSION(iphdr) == IP_HEADER_VERSION_4) {

            if (iphdr->ip_p == IP_PROTO_TCP) {

                *istcp = true;

            } else if (iphdr->ip_p == IP_PROTO_UDP) {

                *isudp = true;

            }

        }

    } else if (proto == ETH_P_IPV6) {

        uint8_t l4proto;

        size_t full_ip6hdr_len;



        struct iovec hdr_vec;

        hdr_vec.iov_base = (void *) headers;

        hdr_vec.iov_len = hdr_length;



        *isip6 = true;

        if (eth_parse_ipv6_hdr(&hdr_vec, 1, l2hdr_len,

                              &l4proto, &full_ip6hdr_len)) {

            if (l4proto == IP_PROTO_TCP) {

                *istcp = true;

            } else if (l4proto == IP_PROTO_UDP) {

                *isudp = true;

            }

        }

    }

}
