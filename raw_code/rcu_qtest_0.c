static void rcu_qtest(const char *test, int duration, int nreaders)

{

    int i;

    long long n_removed_local = 0;



    struct list_element *el, *prev_el;



    rcu_qtest_init();

    for (i = 0; i < nreaders; i++) {

        create_thread(rcu_q_reader);

    }

    create_thread(rcu_q_updater);

    rcu_qtest_run(duration, nreaders);



    QLIST_FOREACH_SAFE_RCU(prev_el, &Q_list_head, entry, el) {

        QLIST_REMOVE_RCU(prev_el, entry);

        call_rcu1(&prev_el->rcu, reclaim_list_el);

        n_removed_local++;

    }

    atomic_add(&n_nodes_removed, n_removed_local);

    synchronize_rcu();

    while (n_nodes_removed > n_reclaims) {

        g_usleep(100);

        synchronize_rcu();

    }

    if (g_test_in_charge) {

        g_assert_cmpint(n_nodes_removed, ==, n_reclaims);

    } else {

        printf("%s: %d readers; 1 updater; nodes read: "  \

               "%lld, nodes removed: %lld; nodes reclaimed: %lld\n",

               test, nthreadsrunning - 1, n_reads, n_nodes_removed, n_reclaims);

        exit(0);

    }

}
