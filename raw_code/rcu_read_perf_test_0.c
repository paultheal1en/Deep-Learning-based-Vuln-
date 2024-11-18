static void *rcu_read_perf_test(void *arg)

{

    int i;

    long long n_reads_local = 0;



    rcu_register_thread();



    *(struct rcu_reader_data **)arg = &rcu_reader;

    atomic_inc(&nthreadsrunning);

    while (goflag == GOFLAG_INIT) {

        g_usleep(1000);

    }

    while (goflag == GOFLAG_RUN) {

        for (i = 0; i < RCU_READ_RUN; i++) {

            rcu_read_lock();

            rcu_read_unlock();

        }

        n_reads_local += RCU_READ_RUN;

    }

    atomic_add(&n_reads, n_reads_local);



    rcu_unregister_thread();

    return NULL;

}
