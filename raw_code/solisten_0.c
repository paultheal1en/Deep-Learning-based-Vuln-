solisten(port, laddr, lport, flags)

	u_int port;

	u_int32_t laddr;

	u_int lport;

	int flags;

{

	struct sockaddr_in addr;

	struct socket *so;

	int s, addrlen = sizeof(addr), opt = 1;



	DEBUG_CALL("solisten");

	DEBUG_ARG("port = %d", port);

	DEBUG_ARG("laddr = %x", laddr);

	DEBUG_ARG("lport = %d", lport);

	DEBUG_ARG("flags = %x", flags);



	if ((so = socreate()) == NULL) {

	  /* free(so);      Not sofree() ??? free(NULL) == NOP */

	  return NULL;

	}



	/* Don't tcp_attach... we don't need so_snd nor so_rcv */

	if ((so->so_tcpcb = tcp_newtcpcb(so)) == NULL) {

		free(so);

		return NULL;

	}

	insque(so,&tcb);



	/*

	 * SS_FACCEPTONCE sockets must time out.

	 */

	if (flags & SS_FACCEPTONCE)

	   so->so_tcpcb->t_timer[TCPT_KEEP] = TCPTV_KEEP_INIT*2;



	so->so_state = (SS_FACCEPTCONN|flags);

	so->so_lport = lport; /* Kept in network format */

	so->so_laddr.s_addr = laddr; /* Ditto */



	addr.sin_family = AF_INET;

	addr.sin_addr.s_addr = INADDR_ANY;

	addr.sin_port = port;



	if (((s = socket(AF_INET,SOCK_STREAM,0)) < 0) ||

	    (setsockopt(s,SOL_SOCKET,SO_REUSEADDR,(char *)&opt,sizeof(int)) < 0) ||

	    (bind(s,(struct sockaddr *)&addr, sizeof(addr)) < 0) ||

	    (listen(s,1) < 0)) {

		int tmperrno = errno; /* Don't clobber the real reason we failed */



		close(s);

		sofree(so);

		/* Restore the real errno */

#ifdef _WIN32

		WSASetLastError(tmperrno);

#else

		errno = tmperrno;

#endif

		return NULL;

	}

	setsockopt(s,SOL_SOCKET,SO_OOBINLINE,(char *)&opt,sizeof(int));



	getsockname(s,(struct sockaddr *)&addr,&addrlen);

	so->so_fport = addr.sin_port;

	if (addr.sin_addr.s_addr == 0 || addr.sin_addr.s_addr == loopback_addr.s_addr)

	   so->so_faddr = alias_addr;

	else

	   so->so_faddr = addr.sin_addr;



	so->s = s;

	return so;

}
