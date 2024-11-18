void fork_end(int child)

{

    mmap_fork_end(child);

    if (child) {

        CPUState *cpu, *next_cpu;

        /* Child processes created by fork() only have a single thread.

           Discard information about the parent threads.  */

        CPU_FOREACH_SAFE(cpu, next_cpu) {

            if (cpu != thread_cpu) {

                QTAILQ_REMOVE(&cpus, thread_cpu, node);

            }

        }

        pending_cpus = 0;

        pthread_mutex_init(&exclusive_lock, NULL);

        pthread_mutex_init(&cpu_list_mutex, NULL);

        pthread_cond_init(&exclusive_cond, NULL);

        pthread_cond_init(&exclusive_resume, NULL);

        pthread_mutex_init(&tcg_ctx.tb_ctx.tb_lock, NULL);

        gdbserver_fork(thread_cpu);

    } else {

        pthread_mutex_unlock(&exclusive_lock);

        pthread_mutex_unlock(&tcg_ctx.tb_ctx.tb_lock);

    }

}
