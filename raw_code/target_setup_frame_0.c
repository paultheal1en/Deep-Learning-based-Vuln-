static void target_setup_frame(int usig, struct target_sigaction *ka,

                               target_siginfo_t *info, target_sigset_t *set,

                               CPUARMState *env)

{

    struct target_rt_sigframe *frame;

    abi_ulong frame_addr, return_addr;



    frame_addr = get_sigframe(ka, env);

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0)) {

        goto give_sigsegv;

    }



    __put_user(0, &frame->uc.tuc_flags);

    __put_user(0, &frame->uc.tuc_link);



    __put_user(target_sigaltstack_used.ss_sp,

                      &frame->uc.tuc_stack.ss_sp);

    __put_user(sas_ss_flags(env->xregs[31]),

                      &frame->uc.tuc_stack.ss_flags);

    __put_user(target_sigaltstack_used.ss_size,

                      &frame->uc.tuc_stack.ss_size);

    target_setup_sigframe(frame, env, set);

    if (ka->sa_flags & TARGET_SA_RESTORER) {

        return_addr = ka->sa_restorer;

    } else {

        /* mov x8,#__NR_rt_sigreturn; svc #0 */

        __put_user(0xd2801168, &frame->tramp[0]);

        __put_user(0xd4000001, &frame->tramp[1]);

        return_addr = frame_addr + offsetof(struct target_rt_sigframe, tramp);

    }

    env->xregs[0] = usig;

    env->xregs[31] = frame_addr;

    env->xregs[29] = env->xregs[31] + offsetof(struct target_rt_sigframe, fp);

    env->pc = ka->_sa_handler;

    env->xregs[30] = return_addr;

    if (info) {

        if (copy_siginfo_to_user(&frame->info, info)) {

            goto give_sigsegv;

        }

        env->xregs[1] = frame_addr + offsetof(struct target_rt_sigframe, info);

        env->xregs[2] = frame_addr + offsetof(struct target_rt_sigframe, uc);

    }



    unlock_user_struct(frame, frame_addr, 1);

    return;



 give_sigsegv:

    unlock_user_struct(frame, frame_addr, 1);

    force_sig(TARGET_SIGSEGV);

}
