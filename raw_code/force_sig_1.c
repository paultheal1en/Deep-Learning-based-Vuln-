static void QEMU_NORETURN force_sig(int target_sig)

{

    CPUState *cpu = thread_cpu;

    CPUArchState *env = cpu->env_ptr;

    TaskState *ts = (TaskState *)cpu->opaque;

    int host_sig, core_dumped = 0;

    struct sigaction act;



    host_sig = target_to_host_signal(target_sig);

    trace_user_force_sig(env, target_sig, host_sig);

    gdb_signalled(env, target_sig);



    /* dump core if supported by target binary format */

    if (core_dump_signal(target_sig) && (ts->bprm->core_dump != NULL)) {

        stop_all_tasks();

        core_dumped =

            ((*ts->bprm->core_dump)(target_sig, env) == 0);

    }

    if (core_dumped) {

        /* we already dumped the core of target process, we don't want

         * a coredump of qemu itself */

        struct rlimit nodump;

        getrlimit(RLIMIT_CORE, &nodump);

        nodump.rlim_cur=0;

        setrlimit(RLIMIT_CORE, &nodump);

        (void) fprintf(stderr, "qemu: uncaught target signal %d (%s) - %s\n",

            target_sig, strsignal(host_sig), "core dumped" );

    }



    /* The proper exit code for dying from an uncaught signal is

     * -<signal>.  The kernel doesn't allow exit() or _exit() to pass

     * a negative value.  To get the proper exit code we need to

     * actually die from an uncaught signal.  Here the default signal

     * handler is installed, we send ourself a signal and we wait for

     * it to arrive. */

    sigfillset(&act.sa_mask);

    act.sa_handler = SIG_DFL;

    act.sa_flags = 0;

    sigaction(host_sig, &act, NULL);



    /* For some reason raise(host_sig) doesn't send the signal when

     * statically linked on x86-64. */

    kill(getpid(), host_sig);



    /* Make sure the signal isn't masked (just reuse the mask inside

    of act) */

    sigdelset(&act.sa_mask, host_sig);

    sigsuspend(&act.sa_mask);



    /* unreachable */

    abort();

}
