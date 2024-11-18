static void setup_frame(int sig, struct target_sigaction *ka,

                        target_sigset_t *set, CPUPPCState *env)

{

    struct target_sigframe *frame;

    struct target_sigcontext *sc;

    target_ulong frame_addr, newsp;

    int err = 0;

    int signal;



    frame_addr = get_sigframe(ka, env, sizeof(*frame));

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 1))

        goto sigsegv;

    sc = &frame->sctx;



    signal = current_exec_domain_sig(sig);



    __put_user(ka->_sa_handler, &sc->handler);

    __put_user(set->sig[0], &sc->oldmask);

#if defined(TARGET_PPC64)

    __put_user(set->sig[0] >> 32, &sc->_unused[3]);

#else

    __put_user(set->sig[1], &sc->_unused[3]);

#endif

    __put_user(h2g(&frame->mctx), &sc->regs);

    __put_user(sig, &sc->signal);



    /* Save user regs.  */

    err |= save_user_regs(env, &frame->mctx, TARGET_NR_sigreturn);



    /* The kernel checks for the presence of a VDSO here.  We don't

       emulate a vdso, so use a sigreturn system call.  */

    env->lr = (target_ulong) h2g(frame->mctx.tramp);



    /* Turn off all fp exceptions.  */

    env->fpscr = 0;



    /* Create a stack frame for the caller of the handler.  */

    newsp = frame_addr - SIGNAL_FRAMESIZE;

    err |= put_user(env->gpr[1], newsp, target_ulong);



    if (err)

        goto sigsegv;



    /* Set up registers for signal handler.  */

    env->gpr[1] = newsp;

    env->gpr[3] = signal;

    env->gpr[4] = frame_addr + offsetof(struct target_sigframe, sctx);

    env->nip = (target_ulong) ka->_sa_handler;

    /* Signal handlers are entered in big-endian mode.  */

    env->msr &= ~MSR_LE;



    unlock_user_struct(frame, frame_addr, 1);

    return;



sigsegv:

    unlock_user_struct(frame, frame_addr, 1);

    qemu_log("segfaulting from setup_frame\n");

    force_sig(TARGET_SIGSEGV);

}
