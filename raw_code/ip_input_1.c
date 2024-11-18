ip_input(struct mbuf *m)

{

	Slirp *slirp = m->slirp;

	register struct ip *ip;

	int hlen;



	DEBUG_CALL("ip_input");

	DEBUG_ARG("m = %p", m);

	DEBUG_ARG("m_len = %d", m->m_len);



	if (m->m_len < sizeof (struct ip)) {

		return;

	}



	ip = mtod(m, struct ip *);



	if (ip->ip_v != IPVERSION) {

		goto bad;

	}



	hlen = ip->ip_hl << 2;

	if (hlen<sizeof(struct ip ) || hlen>m->m_len) {/* min header length */

	  goto bad;                                  /* or packet too short */

	}



        /* keep ip header intact for ICMP reply

	 * ip->ip_sum = cksum(m, hlen);

	 * if (ip->ip_sum) {

	 */

	if(cksum(m,hlen)) {

	  goto bad;

	}



	/*

	 * Convert fields to host representation.

	 */

	NTOHS(ip->ip_len);

	if (ip->ip_len < hlen) {

		goto bad;

	}

	NTOHS(ip->ip_id);

	NTOHS(ip->ip_off);



	/*

	 * Check that the amount of data in the buffers

	 * is as at least much as the IP header would have us expect.

	 * Trim mbufs if longer than we expect.

	 * Drop packet if shorter than we expect.

	 */

	if (m->m_len < ip->ip_len) {

		goto bad;

	}



	/* Should drop packet if mbuf too long? hmmm... */

	if (m->m_len > ip->ip_len)

	   m_adj(m, ip->ip_len - m->m_len);



	/* check ip_ttl for a correct ICMP reply */

	if (ip->ip_ttl == 0) {

	    icmp_send_error(m, ICMP_TIMXCEED, ICMP_TIMXCEED_INTRANS, 0, "ttl");

	    goto bad;

	}



	/*

	 * If offset or IP_MF are set, must reassemble.

	 * Otherwise, nothing need be done.

	 * (We could look in the reassembly queue to see

	 * if the packet was previously fragmented,

	 * but it's not worth the time; just let them time out.)

	 *

	 * XXX This should fail, don't fragment yet

	 */

	if (ip->ip_off &~ IP_DF) {

	  register struct ipq *fp;

      struct qlink *l;

		/*

		 * Look for queue of fragments

		 * of this datagram.

		 */

		for (l = slirp->ipq.ip_link.next; l != &slirp->ipq.ip_link;

		     l = l->next) {

            fp = container_of(l, struct ipq, ip_link);

            if (ip->ip_id == fp->ipq_id &&

                    ip->ip_src.s_addr == fp->ipq_src.s_addr &&

                    ip->ip_dst.s_addr == fp->ipq_dst.s_addr &&

                    ip->ip_p == fp->ipq_p)

		    goto found;

        }

        fp = NULL;

	found:



		/*

		 * Adjust ip_len to not reflect header,

		 * set ip_mff if more fragments are expected,

		 * convert offset of this to bytes.

		 */

		ip->ip_len -= hlen;

		if (ip->ip_off & IP_MF)

		  ip->ip_tos |= 1;

		else

		  ip->ip_tos &= ~1;



		ip->ip_off <<= 3;



		/*

		 * If datagram marked as having more fragments

		 * or if this is not the first fragment,

		 * attempt reassembly; if it succeeds, proceed.

		 */

		if (ip->ip_tos & 1 || ip->ip_off) {

			ip = ip_reass(slirp, ip, fp);

                        if (ip == NULL)

				return;

			m = dtom(slirp, ip);

		} else

			if (fp)

		   	   ip_freef(slirp, fp);



	} else

		ip->ip_len -= hlen;



	/*

	 * Switch out to protocol's input routine.

	 */

	switch (ip->ip_p) {

	 case IPPROTO_TCP:

		tcp_input(m, hlen, (struct socket *)NULL, AF_INET);

		break;

	 case IPPROTO_UDP:

		udp_input(m, hlen);

		break;

	 case IPPROTO_ICMP:

		icmp_input(m, hlen);

		break;

	 default:

		m_free(m);

	}

	return;

bad:

	m_free(m);

}
