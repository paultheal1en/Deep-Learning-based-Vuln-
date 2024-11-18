static void setup_rt_frame(int sig, struct target_sigaction *ka,

                           target_siginfo_t *info,

                           target_sigset_t *set, CPUPPCState *env)

{

    struct target_rt_sigframe *rt_sf;

    struct target_mcontext *frame;

    target_ulong rt_sf_addr, newsp = 0;

    int i, err = 0;

    int signal;



    rt_sf_addr = get_sigframe(ka, env, sizeof(*rt_sf));

    if (!lock_user_struct(VERIFY_WRITE, rt_sf, rt_sf_addr, 1))

        goto sigsegv;



    signal = current_exec_domain_sig(sig);



    copy_siginfo_to_user(&rt_sf->info, info);



    __put_user(0, &rt_sf->uc.tuc_flags);

    __put_user(0, &rt_sf->uc.tuc_link);

    __put_user((target_ulong)target_sigaltstack_used.ss_sp,

               &rt_sf->uc.tuc_stack.ss_sp);

    __put_user(sas_ss_flags(env->gpr[1]),

               &rt_sf->uc.tuc_stack.ss_flags);

    __put_user(target_sigaltstack_used.ss_size,

               &rt_sf->uc.tuc_stack.ss_size);

    __put_user(h2g (&rt_sf->uc.tuc_mcontext),

               &rt_sf->uc.tuc_regs);

    for(i = 0; i < TARGET_NSIG_WORDS; i++) {

        __put_user(set->sig[i], &rt_sf->uc.tuc_sigmask.sig[i]);

    }



    frame = &rt_sf->uc.tuc_mcontext;

    err |= save_user_regs(env, frame, TARGET_NR_rt_sigreturn);



    /* The kernel checks for the presence of a VDSO here.  We don't

       emulate a vdso, so use a sigreturn system call.  */

    env->lr = (target_ulong) h2g(frame->tramp);



    /* Turn off all fp exceptions.  */

    env->fpscr = 0;



    /* Create a stack frame for the caller of the handler.  */

    newsp = rt_sf_addr - (SIGNAL_FRAMESIZE + 16);

    __put_user(env->gpr[1], (target_ulong *)(uintptr_t) newsp);



    if (err)

        goto sigsegv;



    /* Set up registers for signal handler.  */

    env->gpr[1] = newsp;

    env->gpr[3] = (target_ulong) signal;

    env->gpr[4] = (target_ulong) h2g(&rt_sf->info);

    env->gpr[5] = (target_ulong) h2g(&rt_sf->uc);

    env->gpr[6] = (target_ulong) h2g(rt_sf);

    env->nip = (target_ulong) ka->_sa_handler;

    /* Signal handlers are entered in big-endian mode.  */

    env->msr &= ~MSR_LE;



    unlock_user_struct(rt_sf, rt_sf_addr, 1);

    return;



sigsegv:

    unlock_user_struct(rt_sf, rt_sf_addr, 1);

    qemu_log("segfaulting from setup_rt_frame\n");

    force_sig(TARGET_SIGSEGV);



}
