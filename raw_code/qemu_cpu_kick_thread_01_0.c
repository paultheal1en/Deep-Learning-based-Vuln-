static void qemu_cpu_kick_thread(CPUState *cpu)

{

#ifndef _WIN32

    int err;



    err = pthread_kill(cpu->thread->thread, SIG_IPI);

    if (err) {

        fprintf(stderr, "qemu:%s: %s", __func__, strerror(err));

        exit(1);

    }

#else /* _WIN32 */

    if (!qemu_cpu_is_self(cpu)) {

        SuspendThread(cpu->hThread);

        cpu_signal(0);

        ResumeThread(cpu->hThread);

    }

#endif

}
