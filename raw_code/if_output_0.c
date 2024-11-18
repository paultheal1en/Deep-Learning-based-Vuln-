if_output(struct socket *so, struct mbuf *ifm)

{

	struct mbuf *ifq;

	int on_fastq = 1;



	DEBUG_CALL("if_output");

	DEBUG_ARG("so = %lx", (long)so);

	DEBUG_ARG("ifm = %lx", (long)ifm);



	/*

	 * First remove the mbuf from m_usedlist,

	 * since we're gonna use m_next and m_prev ourselves

	 * XXX Shouldn't need this, gotta change dtom() etc.

	 */

	if (ifm->m_flags & M_USEDLIST) {

		remque(ifm);

		ifm->m_flags &= ~M_USEDLIST;

	}



	/*

	 * See if there's already a batchq list for this session.

	 * This can include an interactive session, which should go on fastq,

	 * but gets too greedy... hence it'll be downgraded from fastq to batchq.

	 * We mustn't put this packet back on the fastq (or we'll send it out of order)

	 * XXX add cache here?

	 */

	for (ifq = if_batchq.ifq_prev; ifq != &if_batchq; ifq = ifq->ifq_prev) {

		if (so == ifq->ifq_so) {

			/* A match! */

			ifm->ifq_so = so;

			ifs_insque(ifm, ifq->ifs_prev);

			goto diddit;

		}

	}



	/* No match, check which queue to put it on */

	if (so && (so->so_iptos & IPTOS_LOWDELAY)) {

		ifq = if_fastq.ifq_prev;

		on_fastq = 1;

		/*

		 * Check if this packet is a part of the last

		 * packet's session

		 */

		if (ifq->ifq_so == so) {

			ifm->ifq_so = so;

			ifs_insque(ifm, ifq->ifs_prev);

			goto diddit;

		}

	} else

		ifq = if_batchq.ifq_prev;



	/* Create a new doubly linked list for this session */

	ifm->ifq_so = so;

	ifs_init(ifm);

	insque(ifm, ifq);



diddit:

	++if_queued;



	if (so) {

		/* Update *_queued */

		so->so_queued++;

		so->so_nqueued++;

		/*

		 * Check if the interactive session should be downgraded to

		 * the batchq.  A session is downgraded if it has queued 6

		 * packets without pausing, and at least 3 of those packets

		 * have been sent over the link

		 * (XXX These are arbitrary numbers, probably not optimal..)

		 */

		if (on_fastq && ((so->so_nqueued >= 6) &&

				 (so->so_nqueued - so->so_queued) >= 3)) {



			/* Remove from current queue... */

			remque(ifm->ifs_next);



			/* ...And insert in the new.  That'll teach ya! */

			insque(ifm->ifs_next, &if_batchq);

		}

	}



#ifndef FULL_BOLT

	/*

	 * This prevents us from malloc()ing too many mbufs

	 */

	if (link_up) {

		/* if_start will check towrite */

		if_start();

	}

#endif

}
