static void bootp_reply(Slirp *slirp, const struct bootp_t *bp)

{

    BOOTPClient *bc = NULL;

    struct mbuf *m;

    struct bootp_t *rbp;

    struct sockaddr_in saddr, daddr;

    struct in_addr preq_addr;

    int dhcp_msg_type, val;

    uint8_t *q;

    uint8_t client_ethaddr[ETH_ALEN];



    /* extract exact DHCP msg type */

    dhcp_decode(bp, &dhcp_msg_type, &preq_addr);

    DPRINTF("bootp packet op=%d msgtype=%d", bp->bp_op, dhcp_msg_type);

    if (preq_addr.s_addr != htonl(0L))

        DPRINTF(" req_addr=%08" PRIx32 "\n", ntohl(preq_addr.s_addr));

    else

        DPRINTF("\n");



    if (dhcp_msg_type == 0)

        dhcp_msg_type = DHCPREQUEST; /* Force reply for old BOOTP clients */



    if (dhcp_msg_type != DHCPDISCOVER &&

        dhcp_msg_type != DHCPREQUEST)

        return;



    /* Get client's hardware address from bootp request */

    memcpy(client_ethaddr, bp->bp_hwaddr, ETH_ALEN);



    m = m_get(slirp);

    if (!m) {

        return;

    }

    m->m_data += IF_MAXLINKHDR;

    rbp = (struct bootp_t *)m->m_data;

    m->m_data += sizeof(struct udpiphdr);

    memset(rbp, 0, sizeof(struct bootp_t));



    if (dhcp_msg_type == DHCPDISCOVER) {

        if (preq_addr.s_addr != htonl(0L)) {

            bc = request_addr(slirp, &preq_addr, client_ethaddr);

            if (bc) {

                daddr.sin_addr = preq_addr;

            }

        }

        if (!bc) {

         new_addr:

            bc = get_new_addr(slirp, &daddr.sin_addr, client_ethaddr);

            if (!bc) {

                DPRINTF("no address left\n");

                return;

            }

        }

        memcpy(bc->macaddr, client_ethaddr, ETH_ALEN);

    } else if (preq_addr.s_addr != htonl(0L)) {

        bc = request_addr(slirp, &preq_addr, client_ethaddr);

        if (bc) {

            daddr.sin_addr = preq_addr;

            memcpy(bc->macaddr, client_ethaddr, ETH_ALEN);

        } else {

            /* DHCPNAKs should be sent to broadcast */

            daddr.sin_addr.s_addr = 0xffffffff;

        }

    } else {

        bc = find_addr(slirp, &daddr.sin_addr, bp->bp_hwaddr);

        if (!bc) {

            /* if never assigned, behaves as if it was already

               assigned (windows fix because it remembers its address) */

            goto new_addr;

        }

    }



    /* Update ARP table for this IP address */

    arp_table_add(slirp, daddr.sin_addr.s_addr, client_ethaddr);



    saddr.sin_addr = slirp->vhost_addr;

    saddr.sin_port = htons(BOOTP_SERVER);



    daddr.sin_port = htons(BOOTP_CLIENT);



    rbp->bp_op = BOOTP_REPLY;

    rbp->bp_xid = bp->bp_xid;

    rbp->bp_htype = 1;

    rbp->bp_hlen = 6;

    memcpy(rbp->bp_hwaddr, bp->bp_hwaddr, ETH_ALEN);



    rbp->bp_yiaddr = daddr.sin_addr; /* Client IP address */

    rbp->bp_siaddr = saddr.sin_addr; /* Server IP address */



    q = rbp->bp_vend;

    memcpy(q, rfc1533_cookie, 4);

    q += 4;



    if (bc) {

        DPRINTF("%s addr=%08" PRIx32 "\n",

                (dhcp_msg_type == DHCPDISCOVER) ? "offered" : "ack'ed",

                ntohl(daddr.sin_addr.s_addr));



        if (dhcp_msg_type == DHCPDISCOVER) {

            *q++ = RFC2132_MSG_TYPE;

            *q++ = 1;

            *q++ = DHCPOFFER;

        } else /* DHCPREQUEST */ {

            *q++ = RFC2132_MSG_TYPE;

            *q++ = 1;

            *q++ = DHCPACK;

        }



        if (slirp->bootp_filename)

            snprintf((char *)rbp->bp_file, sizeof(rbp->bp_file), "%s",

                     slirp->bootp_filename);



        *q++ = RFC2132_SRV_ID;

        *q++ = 4;

        memcpy(q, &saddr.sin_addr, 4);

        q += 4;



        *q++ = RFC1533_NETMASK;

        *q++ = 4;

        memcpy(q, &slirp->vnetwork_mask, 4);

        q += 4;



        if (!slirp->restricted) {

            *q++ = RFC1533_GATEWAY;

            *q++ = 4;

            memcpy(q, &saddr.sin_addr, 4);

            q += 4;



            *q++ = RFC1533_DNS;

            *q++ = 4;

            memcpy(q, &slirp->vnameserver_addr, 4);

            q += 4;

        }



        *q++ = RFC2132_LEASE_TIME;

        *q++ = 4;

        val = htonl(LEASE_TIME);

        memcpy(q, &val, 4);

        q += 4;



        if (*slirp->client_hostname) {

            val = strlen(slirp->client_hostname);

            *q++ = RFC1533_HOSTNAME;

            *q++ = val;

            memcpy(q, slirp->client_hostname, val);

            q += val;

        }



        if (slirp->vdnssearch) {

            size_t spaceleft = sizeof(rbp->bp_vend) - (q - rbp->bp_vend);

            val = slirp->vdnssearch_len;

            if (val + 1 > spaceleft) {

                g_warning("DHCP packet size exceeded, "

                    "omitting domain-search option.");

            } else {

                memcpy(q, slirp->vdnssearch, val);

                q += val;

            }

        }

    } else {

        static const char nak_msg[] = "requested address not available";



        DPRINTF("nak'ed addr=%08" PRIx32 "\n", ntohl(preq_addr.s_addr));



        *q++ = RFC2132_MSG_TYPE;

        *q++ = 1;

        *q++ = DHCPNAK;



        *q++ = RFC2132_MESSAGE;

        *q++ = sizeof(nak_msg) - 1;

        memcpy(q, nak_msg, sizeof(nak_msg) - 1);

        q += sizeof(nak_msg) - 1;

    }

    *q = RFC1533_END;



    daddr.sin_addr.s_addr = 0xffffffffu;



    m->m_len = sizeof(struct bootp_t) -

        sizeof(struct ip) - sizeof(struct udphdr);

    udp_output2(NULL, m, &saddr, &daddr, IPTOS_LOWDELAY);

}
