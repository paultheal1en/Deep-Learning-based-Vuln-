void slirp_input(const uint8_t *pkt, int pkt_len)

{

    struct mbuf *m;

    int proto;



    if (pkt_len < ETH_HLEN)

        return;



    proto = ntohs(*(uint16_t *)(pkt + 12));

    switch(proto) {

    case ETH_P_ARP:

        arp_input(pkt, pkt_len);

        break;

    case ETH_P_IP:

        m = m_get();

        if (!m)

            return;

        /* Note: we add to align the IP header */




        m->m_len = pkt_len + 2;

        memcpy(m->m_data + 2, pkt, pkt_len);



        m->m_data += 2 + ETH_HLEN;

        m->m_len -= 2 + ETH_HLEN;



        ip_input(m);

        break;

    default:

        break;

