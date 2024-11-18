void arp_table_add(Slirp *slirp, uint32_t ip_addr, uint8_t ethaddr[ETH_ALEN])

{

    const uint32_t broadcast_addr =

        ~slirp->vnetwork_mask.s_addr | slirp->vnetwork_addr.s_addr;

    ArpTable *arptbl = &slirp->arp_table;

    int i;



    DEBUG_CALL("arp_table_add");

    DEBUG_ARG("ip = 0x%x", ip_addr);

    DEBUG_ARGS((dfd, " hw addr = %02x:%02x:%02x:%02x:%02x:%02x\n",

                ethaddr[0], ethaddr[1], ethaddr[2],

                ethaddr[3], ethaddr[4], ethaddr[5]));



    /* Check 0.0.0.0/8 invalid source-only addresses */

    assert((ip_addr & htonl(~(0xf << 28))) != 0);



    if (ip_addr == 0xffffffff || ip_addr == broadcast_addr) {

        /* Do not register broadcast addresses */

        return;

    }



    /* Search for an entry */

    for (i = 0; i < ARP_TABLE_SIZE; i++) {

        if (arptbl->table[i].ar_sip == ip_addr) {

            /* Update the entry */

            memcpy(arptbl->table[i].ar_sha, ethaddr, ETH_ALEN);

            return;

        }

    }



    /* No entry found, create a new one */

    arptbl->table[arptbl->next_victim].ar_sip = ip_addr;

    memcpy(arptbl->table[arptbl->next_victim].ar_sha,  ethaddr, ETH_ALEN);

    arptbl->next_victim = (arptbl->next_victim + 1) % ARP_TABLE_SIZE;

}
