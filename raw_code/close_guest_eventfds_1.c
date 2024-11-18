static void close_guest_eventfds(IVShmemState *s, int posn)

{

    int i, guest_curr_max;



    if (!ivshmem_has_feature(s, IVSHMEM_IOEVENTFD)) {








    guest_curr_max = s->peers[posn].nb_eventfds;



    memory_region_transaction_begin();

    for (i = 0; i < guest_curr_max; i++) {

        ivshmem_del_eventfd(s, posn, i);


    memory_region_transaction_commit();

    for (i = 0; i < guest_curr_max; i++) {

        event_notifier_cleanup(&s->peers[posn].eventfds[i]);




    g_free(s->peers[posn].eventfds);

    s->peers[posn].nb_eventfds = 0;
