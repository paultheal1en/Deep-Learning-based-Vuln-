static int tftp_send_oack(struct tftp_session *spt,

                          const char *keys[], uint32_t values[], int nb,

                          struct tftp_t *recv_tp)

{

    struct sockaddr_in saddr, daddr;

    struct mbuf *m;

    struct tftp_t *tp;

    int i, n = 0;



    m = m_get(spt->slirp);



    if (!m)

	return -1;



    memset(m->m_data, 0, m->m_size);



    m->m_data += IF_MAXLINKHDR;

    tp = (void *)m->m_data;

    m->m_data += sizeof(struct udpiphdr);



    tp->tp_op = htons(TFTP_OACK);

    for (i = 0; i < nb; i++) {

        n += snprintf(tp->x.tp_buf + n, sizeof(tp->x.tp_buf) - n, "%s",

                      keys[i]) + 1;

        n += snprintf(tp->x.tp_buf + n, sizeof(tp->x.tp_buf) - n, "%u",

                      values[i]) + 1;

    }



    saddr.sin_addr = recv_tp->ip.ip_dst;

    saddr.sin_port = recv_tp->udp.uh_dport;



    daddr.sin_addr = spt->client_ip;

    daddr.sin_port = spt->client_port;



    m->m_len = sizeof(struct tftp_t) - 514 + n -

        sizeof(struct ip) - sizeof(struct udphdr);

    udp_output2(NULL, m, &saddr, &daddr, IPTOS_LOWDELAY);



    return 0;

}
