sofcantsendmore(struct socket *so)

{

	if ((so->so_state & SS_NOFDREF) == 0) {

            shutdown(so->s,1);           /* send FIN to fhost */

            if (global_readfds) {

                FD_CLR(so->s,global_readfds);

            }

            if (global_xfds) {

                FD_CLR(so->s,global_xfds);

            }

	}

	so->so_state &= ~(SS_ISFCONNECTING);

	if (so->so_state & SS_FCANTRCVMORE) {

	   so->so_state &= SS_PERSISTENT_MASK;

	   so->so_state |= SS_NOFDREF; /* as above */

	} else {

	   so->so_state |= SS_FCANTSENDMORE;

	}

}
