static void *rcu_q_reader(void *arg)

{

    long long j, n_reads_local = 0;

    struct list_element *el;



    *(struct rcu_reader_data **)arg = &rcu_reader;

    atomic_inc(&nthreadsrunning);

    while (goflag == GOFLAG_INIT) {

        g_usleep(1000);

    }



    while (goflag == GOFLAG_RUN) {

        rcu_read_lock();

        QLIST_FOREACH_RCU(el, &Q_list_head, entry) {

            j = atomic_read(&el->val);

            (void)j;

            n_reads_local++;

            if (goflag == GOFLAG_STOP) {

                break;

            }

        }

        rcu_read_unlock();



        g_usleep(100);

    }

    atomic_add(&n_reads, n_reads_local);

    return NULL;

}
