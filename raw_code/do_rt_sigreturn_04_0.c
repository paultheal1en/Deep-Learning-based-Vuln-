long do_rt_sigreturn(CPUM68KState *env)

{

    struct target_rt_sigframe *frame;

    abi_ulong frame_addr = env->aregs[7] - 4;

    target_sigset_t target_set;

    sigset_t set;

    int d0;



    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1))

        goto badframe;



    target_to_host_sigset_internal(&set, &target_set);

    sigprocmask(SIG_SETMASK, &set, NULL);



    /* restore registers */



    if (target_rt_restore_ucontext(env, &frame->uc, &d0))

        goto badframe;



    if (do_sigaltstack(frame_addr +

                       offsetof(struct target_rt_sigframe, uc.tuc_stack),

                       0, get_sp_from_cpustate(env)) == -EFAULT)

        goto badframe;



    unlock_user_struct(frame, frame_addr, 0);

    return d0;



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

    return 0;

}
