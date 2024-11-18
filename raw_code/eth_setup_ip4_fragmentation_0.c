eth_setup_ip4_fragmentation(const void *l2hdr, size_t l2hdr_len,

                            void *l3hdr, size_t l3hdr_len,

                            size_t l3payload_len,

                            size_t frag_offset, bool more_frags)

{

    if (eth_get_l3_proto(l2hdr, l2hdr_len) == ETH_P_IP) {

        uint16_t orig_flags;

        struct ip_header *iphdr = (struct ip_header *) l3hdr;

        uint16_t frag_off_units = frag_offset / IP_FRAG_UNIT_SIZE;

        uint16_t new_ip_off;



        assert(frag_offset % IP_FRAG_UNIT_SIZE == 0);

        assert((frag_off_units & ~IP_OFFMASK) == 0);



        orig_flags = be16_to_cpu(iphdr->ip_off) & ~(IP_OFFMASK|IP_MF);

        new_ip_off = frag_off_units | orig_flags  | (more_frags ? IP_MF : 0);

        iphdr->ip_off = cpu_to_be16(new_ip_off);

        iphdr->ip_len = cpu_to_be16(l3payload_len + l3hdr_len);

    }

}
