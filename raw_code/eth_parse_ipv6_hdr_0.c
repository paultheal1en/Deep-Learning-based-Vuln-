bool eth_parse_ipv6_hdr(struct iovec *pkt, int pkt_frags,

                        size_t ip6hdr_off, uint8_t *l4proto,

                        size_t *full_hdr_len)

{

    struct ip6_header ip6_hdr;

    struct ip6_ext_hdr ext_hdr;

    size_t bytes_read;



    bytes_read = iov_to_buf(pkt, pkt_frags, ip6hdr_off,

                            &ip6_hdr, sizeof(ip6_hdr));

    if (bytes_read < sizeof(ip6_hdr)) {

        return false;

    }



    *full_hdr_len = sizeof(struct ip6_header);



    if (!eth_is_ip6_extension_header_type(ip6_hdr.ip6_nxt)) {

        *l4proto = ip6_hdr.ip6_nxt;

        return true;

    }



    do {

        bytes_read = iov_to_buf(pkt, pkt_frags, ip6hdr_off + *full_hdr_len,

                                &ext_hdr, sizeof(ext_hdr));

        *full_hdr_len += (ext_hdr.ip6r_len + 1) * IP6_EXT_GRANULARITY;

    } while (eth_is_ip6_extension_header_type(ext_hdr.ip6r_nxt));



    *l4proto = ext_hdr.ip6r_nxt;

    return true;

}
