eth_calc_pseudo_hdr_csum(struct ip_header *iphdr, uint16_t csl)

{

    struct ip_pseudo_header ipph;

    ipph.ip_src = iphdr->ip_src;

    ipph.ip_dst = iphdr->ip_dst;

    ipph.ip_payload = cpu_to_be16(csl);

    ipph.ip_proto = iphdr->ip_p;

    ipph.zeros = 0;

    return net_checksum_add(sizeof(ipph), (uint8_t *) &ipph);

}
