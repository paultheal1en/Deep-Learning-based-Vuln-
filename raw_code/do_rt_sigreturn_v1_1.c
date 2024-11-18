static long do_rt_sigreturn_v1(CPUARMState *env)

{

    abi_ulong frame_addr;

    struct rt_sigframe_v1 *frame = NULL;

    sigset_t host_set;



    /*

     * Since we stacked the signal on a 64-bit boundary,

     * then 'sp' should be word aligned here.  If it's

     * not, then the user is trying to mess with us.

     */

    frame_addr = env->regs[13];

    trace_user_do_rt_sigreturn(env, frame_addr);

    if (frame_addr & 7) {

        goto badframe;

    }



    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1)) {

        goto badframe;

    }



    target_to_host_sigset(&host_set, &frame->uc.tuc_sigmask);

    set_sigmask(&host_set);



    if (restore_sigcontext(env, &frame->uc.tuc_mcontext)) {

        goto badframe;

    }



    if (do_sigaltstack(frame_addr + offsetof(struct rt_sigframe_v1, uc.tuc_stack), 0, get_sp_from_cpustate(env)) == -EFAULT)

        goto badframe;



#if 0

    /* Send SIGTRAP if we're single-stepping */

    if (ptrace_cancel_bpt(current))

        send_sig(SIGTRAP, current, 1);

#endif

    unlock_user_struct(frame, frame_addr, 0);

    return -TARGET_QEMU_ESIGRETURN;



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV /* , current */);

    return 0;

}
