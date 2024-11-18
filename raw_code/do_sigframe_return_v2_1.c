static int do_sigframe_return_v2(CPUState *env, target_ulong frame_addr,
                                 struct target_ucontext_v2 *uc)
{
    sigset_t host_set;
    abi_ulong *regspace;
    target_to_host_sigset(&host_set, &uc->tuc_sigmask);
    sigprocmask(SIG_SETMASK, &host_set, NULL);
    if (restore_sigcontext(env, &uc->tuc_mcontext))
    if (do_sigaltstack(frame_addr + offsetof(struct target_ucontext_v2, tuc_stack), 0, get_sp_from_cpustate(env)) == -EFAULT)
#if 0
    /* Send SIGTRAP if we're single-stepping */
    if (ptrace_cancel_bpt(current))
            send_sig(SIGTRAP, current, 1);
#endif
    return 0;