void if_start(Slirp *slirp)

{

    uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

    bool from_batchq, next_from_batchq;

    struct mbuf *ifm, *ifm_next, *ifqt;



    DEBUG_CALL("if_start");



    if (slirp->if_start_busy) {

        return;

    }

    slirp->if_start_busy = true;



    if (slirp->if_fastq.ifq_next != &slirp->if_fastq) {

        ifm_next = slirp->if_fastq.ifq_next;

        next_from_batchq = false;

    } else if (slirp->next_m != &slirp->if_batchq) {

        /* Nothing on fastq, pick up from batchq via next_m */

        ifm_next = slirp->next_m;

        next_from_batchq = true;

    } else {

        ifm_next = NULL;

    }



    while (ifm_next) {

        ifm = ifm_next;

        from_batchq = next_from_batchq;



        ifm_next = ifm->ifq_next;

        if (ifm_next == &slirp->if_fastq) {

            /* No more packets in fastq, switch to batchq */

            ifm_next = slirp->next_m;

            next_from_batchq = true;

        }

        if (ifm_next == &slirp->if_batchq) {

            /* end of batchq */

            ifm_next = NULL;

        }



        /* Try to send packet unless it already expired */

        if (ifm->expiration_date >= now && !if_encap(slirp, ifm)) {

            /* Packet is delayed due to pending ARP resolution */

            continue;

        }



        if (ifm == slirp->next_m) {

            /* Set which packet to send on next iteration */

            slirp->next_m = ifm->ifq_next;

        }



        /* Remove it from the queue */

        ifqt = ifm->ifq_prev;

        remque(ifm);



        /* If there are more packets for this session, re-queue them */

        if (ifm->ifs_next != ifm) {

            struct mbuf *next = ifm->ifs_next;



            insque(next, ifqt);

            ifs_remque(ifm);



            if (!from_batchq) {

                /* Next packet in fastq is from the same session */

                ifm_next = next;

                next_from_batchq = false;

            } else if (slirp->next_m == &slirp->if_batchq) {

                /* Set next_m and ifm_next if the session packet is now the

                 * only one on batchq */

                slirp->next_m = ifm_next = next;

            }

        }



        /* Update so_queued */

        if (ifm->ifq_so && --ifm->ifq_so->so_queued == 0) {

            /* If there's no more queued, reset nqueued */

            ifm->ifq_so->so_nqueued = 0;

        }



        m_free(ifm);

    }



    slirp->if_start_busy = false;

}
