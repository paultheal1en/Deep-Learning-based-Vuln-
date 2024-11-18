static void *thread_func(void *p)

{

    struct thread_info *info = p;



    rcu_register_thread();



    atomic_inc(&n_ready_threads);

    while (!atomic_mb_read(&test_start)) {

        cpu_relax();

    }



    rcu_read_lock();

    while (!atomic_read(&test_stop)) {

        info->r = xorshift64star(info->r);

        info->func(info);

    }

    rcu_read_unlock();



    rcu_unregister_thread();

    return NULL;

}
