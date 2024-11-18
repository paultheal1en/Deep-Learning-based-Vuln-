static void *tcg_cpu_thread_fn(void *arg)

{

    CPUState *env = arg;



    qemu_tcg_init_cpu_signals();

    qemu_thread_self(env->thread);



    /* signal CPU creation */

    qemu_mutex_lock(&qemu_global_mutex);

    for (env = first_cpu; env != NULL; env = env->next_cpu)

        env->created = 1;

    qemu_cond_signal(&qemu_cpu_cond);



    /* and wait for machine initialization */

    while (!qemu_system_ready)

        qemu_cond_timedwait(&qemu_system_cond, &qemu_global_mutex, 100);



    while (1) {

        cpu_exec_all();

        qemu_tcg_wait_io_event();

    }



    return NULL;

}
