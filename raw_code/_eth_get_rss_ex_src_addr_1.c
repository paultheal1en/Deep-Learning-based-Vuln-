_eth_get_rss_ex_src_addr(const struct iovec *pkt, int pkt_frags,

                        size_t dsthdr_offset,

                        struct ip6_ext_hdr *ext_hdr,

                        struct in6_address *src_addr)

{

    size_t bytes_left = (ext_hdr->ip6r_len + 1) * 8 - sizeof(*ext_hdr);

    struct ip6_option_hdr opthdr;

    size_t opt_offset = dsthdr_offset + sizeof(*ext_hdr);



    while (bytes_left > sizeof(opthdr)) {

        size_t input_size = iov_size(pkt, pkt_frags);

        size_t bytes_read, optlen;



        if (input_size < opt_offset) {

            return false;

        }



        bytes_read = iov_to_buf(pkt, pkt_frags, opt_offset,

                                &opthdr, sizeof(opthdr));



        if (bytes_read != sizeof(opthdr)) {

            return false;

        }



        optlen = (opthdr.type == IP6_OPT_PAD1) ? 1

                                               : (opthdr.len + sizeof(opthdr));



        if (optlen > bytes_left) {

            return false;

        }



        if (opthdr.type == IP6_OPT_HOME) {

            size_t input_size = iov_size(pkt, pkt_frags);



            if (input_size < opt_offset + sizeof(opthdr)) {

                return false;

            }



            bytes_read = iov_to_buf(pkt, pkt_frags,

                                    opt_offset + sizeof(opthdr),

                                    src_addr, sizeof(*src_addr));



            return bytes_read == sizeof(src_addr);

        }



        opt_offset += optlen;

        bytes_left -= optlen;

    }



    return false;

}
