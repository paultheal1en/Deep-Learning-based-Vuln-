long do_sigreturn(CPUMBState *env)

{

    struct target_signal_frame *frame;

    abi_ulong frame_addr;

    target_sigset_t target_set;

    sigset_t set;

    int i;



    frame_addr = env->regs[R_SP];

    /* Make sure the guest isn't playing games.  */

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 1))

        goto badframe;



    /* Restore blocked signals */

    if (__get_user(target_set.sig[0], &frame->uc.tuc_mcontext.oldmask))

        goto badframe;

    for(i = 1; i < TARGET_NSIG_WORDS; i++) {

        if (__get_user(target_set.sig[i], &frame->extramask[i - 1]))

            goto badframe;

    }

    target_to_host_sigset_internal(&set, &target_set);

    sigprocmask(SIG_SETMASK, &set, NULL);



    restore_sigcontext(&frame->uc.tuc_mcontext, env);

    /* We got here through a sigreturn syscall, our path back is via an

       rtb insn so setup r14 for that.  */

    env->regs[14] = env->sregs[SR_PC];

 

    unlock_user_struct(frame, frame_addr, 0);

    return env->regs[10];

  badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

}
