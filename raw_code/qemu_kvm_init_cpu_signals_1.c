static void qemu_kvm_init_cpu_signals(CPUState *env)

{

    int r;

    sigset_t set;

    struct sigaction sigact;



    memset(&sigact, 0, sizeof(sigact));

    sigact.sa_handler = dummy_signal;

    sigaction(SIG_IPI, &sigact, NULL);



#ifdef CONFIG_IOTHREAD

    pthread_sigmask(SIG_BLOCK, NULL, &set);

    sigdelset(&set, SIG_IPI);

    sigdelset(&set, SIGBUS);

    r = kvm_set_signal_mask(env, &set);

    if (r) {

        fprintf(stderr, "kvm_set_signal_mask: %s\n", strerror(-r));

        exit(1);

    }

#else

    sigemptyset(&set);

    sigaddset(&set, SIG_IPI);

    sigaddset(&set, SIGIO);

    sigaddset(&set, SIGALRM);

    pthread_sigmask(SIG_BLOCK, &set, NULL);



    pthread_sigmask(SIG_BLOCK, NULL, &set);

    sigdelset(&set, SIGIO);

    sigdelset(&set, SIGALRM);

#endif

    sigdelset(&set, SIG_IPI);

    sigdelset(&set, SIGBUS);

    r = kvm_set_signal_mask(env, &set);

    if (r) {

        fprintf(stderr, "kvm_set_signal_mask: %s\n", strerror(-r));

        exit(1);

    }

}
