void eth_setup_vlan_headers(struct eth_header *ehdr, uint16_t vlan_tag,

    bool *is_new)

{

    struct vlan_header *vhdr = PKT_GET_VLAN_HDR(ehdr);



    switch (be16_to_cpu(ehdr->h_proto)) {

    case ETH_P_VLAN:

    case ETH_P_DVLAN:

        /* vlan hdr exists */

        *is_new = false;

        break;



    default:

        /* No VLAN header, put a new one */

        vhdr->h_proto = ehdr->h_proto;

        ehdr->h_proto = cpu_to_be16(ETH_P_VLAN);

        *is_new = true;

        break;

    }

    vhdr->h_tci = cpu_to_be16(vlan_tag);

}
