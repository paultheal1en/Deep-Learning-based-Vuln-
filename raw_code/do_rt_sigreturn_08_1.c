long do_rt_sigreturn(CPUM68KState *env)

{

    struct target_rt_sigframe *frame;

    abi_ulong frame_addr = env->aregs[7] - 4;

    target_sigset_t target_set;

    sigset_t set;



    trace_user_do_rt_sigreturn(env, frame_addr);

    if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1))

        goto badframe;



    target_to_host_sigset_internal(&set, &target_set);

    set_sigmask(&set);



    /* restore registers */



    if (target_rt_restore_ucontext(env, &frame->uc))

        goto badframe;



    if (do_sigaltstack(frame_addr +

                       offsetof(struct target_rt_sigframe, uc.tuc_stack),

                       0, get_sp_from_cpustate(env)) == -EFAULT)

        goto badframe;



    unlock_user_struct(frame, frame_addr, 0);

    return -TARGET_QEMU_ESIGRETURN;



badframe:

    unlock_user_struct(frame, frame_addr, 0);

    force_sig(TARGET_SIGSEGV);

    return 0;

}
