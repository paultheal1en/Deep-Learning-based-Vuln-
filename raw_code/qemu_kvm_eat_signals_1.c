static void qemu_kvm_eat_signals(CPUState *env)

{

    struct timespec ts = { 0, 0 };

    siginfo_t siginfo;

    sigset_t waitset;

    sigset_t chkset;

    int r;



    sigemptyset(&waitset);

    sigaddset(&waitset, SIG_IPI);

    sigaddset(&waitset, SIGBUS);



    do {

        r = sigtimedwait(&waitset, &siginfo, &ts);

        if (r == -1 && !(errno == EAGAIN || errno == EINTR)) {

            perror("sigtimedwait");

            exit(1);

        }



        switch (r) {

        case SIGBUS:

            if (kvm_on_sigbus_vcpu(env, siginfo.si_code, siginfo.si_addr)) {

                sigbus_reraise();

            }

            break;

        default:

            break;

        }



        r = sigpending(&chkset);

        if (r == -1) {

            perror("sigpending");

            exit(1);

        }

    } while (sigismember(&chkset, SIG_IPI) || sigismember(&chkset, SIGBUS));



#ifndef CONFIG_IOTHREAD

    if (sigismember(&chkset, SIGIO) || sigismember(&chkset, SIGALRM)) {

        qemu_notify_event();

    }

#endif

}
