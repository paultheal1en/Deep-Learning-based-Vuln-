sofcantrcvmore(struct socket *so)

{

	if ((so->so_state & SS_NOFDREF) == 0) {

		shutdown(so->s,0);

		if(global_writefds) {

		  FD_CLR(so->s,global_writefds);

		}

	}

	so->so_state &= ~(SS_ISFCONNECTING);

	if (so->so_state & SS_FCANTSENDMORE) {

	   so->so_state &= SS_PERSISTENT_MASK;

	   so->so_state |= SS_NOFDREF; /* Don't select it */

	} else {

	   so->so_state |= SS_FCANTRCVMORE;

	}

}
