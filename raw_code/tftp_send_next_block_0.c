static void tftp_send_next_block(struct tftp_session *spt,

                                 struct tftp_t *recv_tp)

{

  struct sockaddr_in saddr, daddr;

  struct mbuf *m;

  struct tftp_t *tp;

  int nobytes;



  m = m_get(spt->slirp);



  if (!m) {

    return;

  }



  memset(m->m_data, 0, m->m_size);



  m->m_data += IF_MAXLINKHDR;

  tp = (void *)m->m_data;

  m->m_data += sizeof(struct udpiphdr);



  tp->tp_op = htons(TFTP_DATA);

  tp->x.tp_data.tp_block_nr = htons((spt->block_nr + 1) & 0xffff);



  saddr.sin_addr = recv_tp->ip.ip_dst;

  saddr.sin_port = recv_tp->udp.uh_dport;



  daddr.sin_addr = spt->client_ip;

  daddr.sin_port = spt->client_port;



  nobytes = tftp_read_data(spt, spt->block_nr, tp->x.tp_data.tp_buf, 512);



  if (nobytes < 0) {

    m_free(m);



    /* send "file not found" error back */



    tftp_send_error(spt, 1, "File not found", tp);



    return;

  }



  m->m_len = sizeof(struct tftp_t) - (512 - nobytes) -

        sizeof(struct ip) - sizeof(struct udphdr);



  udp_output2(NULL, m, &saddr, &daddr, IPTOS_LOWDELAY);



  if (nobytes == 512) {

    tftp_session_update(spt);

  }

  else {

    tftp_session_terminate(spt);

  }



  spt->block_nr++;

}
