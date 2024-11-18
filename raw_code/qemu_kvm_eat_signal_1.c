static void qemu_kvm_eat_signal(CPUState *env, int timeout)

{

    struct timespec ts;

    int r, e;

    siginfo_t siginfo;

    sigset_t waitset;



    ts.tv_sec = timeout / 1000;

    ts.tv_nsec = (timeout % 1000) * 1000000;



    sigemptyset(&waitset);

    sigaddset(&waitset, SIG_IPI);



    qemu_mutex_unlock(&qemu_global_mutex);

    r = sigtimedwait(&waitset, &siginfo, &ts);

    e = errno;

    qemu_mutex_lock(&qemu_global_mutex);



    if (r == -1 && !(e == EAGAIN || e == EINTR)) {

        fprintf(stderr, "sigtimedwait: %s\n", strerror(e));

        exit(1);

    }

}
