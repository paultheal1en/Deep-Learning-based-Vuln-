static void *rcu_read_stress_test(void *arg)

{

    int i;

    int itercnt = 0;

    struct rcu_stress *p;

    int pc;

    long long n_reads_local = 0;

    volatile int garbage = 0;



    rcu_register_thread();



    *(struct rcu_reader_data **)arg = &rcu_reader;

    while (goflag == GOFLAG_INIT) {

        g_usleep(1000);

    }

    while (goflag == GOFLAG_RUN) {

        rcu_read_lock();

        p = atomic_rcu_read(&rcu_stress_current);

        if (p->mbtest == 0) {

            n_mberror++;

        }

        rcu_read_lock();

        for (i = 0; i < 100; i++) {

            garbage++;

        }

        rcu_read_unlock();

        pc = p->pipe_count;

        rcu_read_unlock();

        if ((pc > RCU_STRESS_PIPE_LEN) || (pc < 0)) {

            pc = RCU_STRESS_PIPE_LEN;

        }

        atomic_inc(&rcu_stress_count[pc]);

        n_reads_local++;

        if ((++itercnt % 0x1000) == 0) {

            synchronize_rcu();

        }

    }

    atomic_add(&n_reads, n_reads_local);



    rcu_unregister_thread();

    return NULL;

}
