static void pause_all_vcpus(void)

{

    CPUState *penv = first_cpu;



    while (penv) {

        penv->stop = 1;

        qemu_thread_signal(penv->thread, SIGUSR1);

        qemu_cpu_kick(penv);

        penv = (CPUState *)penv->next_cpu;

    }



    while (!all_vcpus_paused()) {

        qemu_cond_timedwait(&qemu_pause_cond, &qemu_global_mutex, 100);

        penv = first_cpu;

        while (penv) {

            qemu_thread_signal(penv->thread, SIGUSR1);

            penv = (CPUState *)penv->next_cpu;

        }

    }

}
