static void close_peer_eventfds(IVShmemState *s, int posn)

{

    int i, n;



    if (!ivshmem_has_feature(s, IVSHMEM_IOEVENTFD)) {

        return;

    }

    if (posn < 0 || posn >= s->nb_peers) {

        error_report("invalid peer %d", posn);

        return;

    }



    n = s->peers[posn].nb_eventfds;



    memory_region_transaction_begin();

    for (i = 0; i < n; i++) {

        ivshmem_del_eventfd(s, posn, i);

    }

    memory_region_transaction_commit();

    for (i = 0; i < n; i++) {

        event_notifier_cleanup(&s->peers[posn].eventfds[i]);

    }



    g_free(s->peers[posn].eventfds);

    s->peers[posn].nb_eventfds = 0;

}
