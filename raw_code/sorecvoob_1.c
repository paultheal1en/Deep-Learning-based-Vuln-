sorecvoob(struct socket *so)

{

	struct tcpcb *tp = sototcpcb(so);



	DEBUG_CALL("sorecvoob");

	DEBUG_ARG("so = %p", so);



	/*

	 * We take a guess at how much urgent data has arrived.

	 * In most situations, when urgent data arrives, the next

	 * read() should get all the urgent data.  This guess will

	 * be wrong however if more data arrives just after the

	 * urgent data, or the read() doesn't return all the

	 * urgent data.

	 */

	soread(so);

	tp->snd_up = tp->snd_una + so->so_snd.sb_cc;

	tp->t_force = 1;

	tcp_output(tp);

	tp->t_force = 0;

}
