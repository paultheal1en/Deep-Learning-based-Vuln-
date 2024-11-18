static long do_sigreturn_v1(CPUARMState *env)

{

    abi_ulong frame_addr;

    struct sigframe_v1 *frame = NULL;

    target_sigset_t set;

    sigset_t host_set;

    int i;



    /*

     * Since we stacked the signal on a 64-bit boundary,

     * then 'sp' should be word aligned here.  If it's

     * not, then the user is trying to mess with us.

     */

    frame_addr = env->regs[13];

    trace_user_do_sigreturn(env, frame_addr);

    if (frame_addr & 7) {

        goto badframe;

    }



    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }



    __get_user(set.sig[0], &frame->sc.oldmask);

    for(i = 1; i < TARGET_NSIG_WORDS; i++) {

        __get_user(set.sig[i], &frame->extramask[i - 1]);

    }



    target_to_host_sigset_internal(&host_set, &set);

    do_sigprocmask(SIG_SETMASK, &host_set, NULL);



    if (restore_sigcontext(env, &frame->sc)) {

        goto badframe;

    }



#if 0

    /* Send SIGTRAP if we're single-stepping */

    if (ptrace_cancel_bpt(current))

        send_sig(SIGTRAP, current, 1);

#endif

    unlock_user_struct(frame, frame_addr, 0);

    return env->regs[0];



badframe:

    force_sig(TARGET_SIGSEGV /* , current */);

    return 0;

}
