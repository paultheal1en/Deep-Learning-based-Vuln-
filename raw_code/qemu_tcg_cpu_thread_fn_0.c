static void *qemu_tcg_cpu_thread_fn(void *arg)

{

    CPUState *cpu = arg;



    qemu_tcg_init_cpu_signals();

    qemu_thread_get_self(cpu->thread);



    qemu_mutex_lock(&qemu_global_mutex);

    CPU_FOREACH(cpu) {

        cpu->thread_id = qemu_get_thread_id();

        cpu->created = true;

        cpu->exception_index = -1;

        cpu->can_do_io = 1;

    }

    qemu_cond_signal(&qemu_cpu_cond);



    /* wait for initial kick-off after machine start */

    while (QTAILQ_FIRST(&cpus)->stopped) {

        qemu_cond_wait(tcg_halt_cond, &qemu_global_mutex);



        /* process any pending work */

        CPU_FOREACH(cpu) {

            qemu_wait_io_event_common(cpu);

        }

    }



    while (1) {

        tcg_exec_all();



        if (use_icount) {

            int64_t deadline = qemu_clock_deadline_ns_all(QEMU_CLOCK_VIRTUAL);



            if (deadline == 0) {

                qemu_clock_notify(QEMU_CLOCK_VIRTUAL);

            }

        }

        qemu_tcg_wait_io_event();

    }



    return NULL;

}
