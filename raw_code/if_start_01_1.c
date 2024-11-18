if_start(Slirp *slirp)

{

    uint64_t now = qemu_get_clock_ns(rt_clock);

    int requeued = 0;

	struct mbuf *ifm, *ifqt;



	DEBUG_CALL("if_start");



	if (slirp->if_queued == 0)

	   return; /* Nothing to do */



 again:

        /* check if we can really output */

        if (!slirp_can_output(slirp->opaque))

            return;



	/*

	 * See which queue to get next packet from

	 * If there's something in the fastq, select it immediately

	 */

	if (slirp->if_fastq.ifq_next != &slirp->if_fastq) {

		ifm = slirp->if_fastq.ifq_next;

	} else {

		/* Nothing on fastq, see if next_m is valid */

		if (slirp->next_m != &slirp->if_batchq)

		   ifm = slirp->next_m;

		else

		   ifm = slirp->if_batchq.ifq_next;



		/* Set which packet to send on next iteration */

		slirp->next_m = ifm->ifq_next;

	}

	/* Remove it from the queue */

	ifqt = ifm->ifq_prev;

	remque(ifm);

	slirp->if_queued--;



	/* If there are more packets for this session, re-queue them */

	if (ifm->ifs_next != /* ifm->ifs_prev != */ ifm) {

		insque(ifm->ifs_next, ifqt);

		ifs_remque(ifm);

	}



	/* Update so_queued */

	if (ifm->ifq_so) {

		if (--ifm->ifq_so->so_queued == 0)

		   /* If there's no more queued, reset nqueued */

		   ifm->ifq_so->so_nqueued = 0;

	}



        if (ifm->expiration_date < now) {

            /* Expired */

            m_free(ifm);

        } else {

            /* Encapsulate the packet for sending */

            if (if_encap(slirp, ifm)) {

                m_free(ifm);

            } else {

                /* re-queue */

                insque(ifm, ifqt);

                requeued++;

            }

        }



	if (slirp->if_queued)

	   goto again;



        slirp->if_queued = requeued;

}
