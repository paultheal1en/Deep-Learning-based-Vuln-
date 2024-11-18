void fork_end(int child)

{

    mmap_fork_end(child);

    if (child) {

        CPUState *cpu, *next_cpu;

        /* Child processes created by fork() only have a single thread.

           Discard information about the parent threads.  */

        CPU_FOREACH_SAFE(cpu, next_cpu) {

            if (cpu != thread_cpu) {

                QTAILQ_REMOVE(&cpus, cpu, node);

            }

        }

        qemu_mutex_init(&tb_ctx.tb_lock);

        qemu_init_cpu_list();

        gdbserver_fork(thread_cpu);




    } else {

        qemu_mutex_unlock(&tb_ctx.tb_lock);

        cpu_list_unlock();


    }

}