static void setup_rt_frame(int sig, struct target_sigaction *ka,

                           target_siginfo_t *info,

                           target_sigset_t *set, CPUAlphaState *env)

{

    abi_ulong frame_addr, r26;

    struct target_rt_sigframe *frame;

    int i, err = 0;



    frame_addr = get_sigframe(ka, env, sizeof(*frame));

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0)) {

        goto give_sigsegv;

    }



    err |= copy_siginfo_to_user(&frame->info, info);



    __put_user(0, &frame->uc.tuc_flags);

    __put_user(0, &frame->uc.tuc_link);

    __put_user(set->sig[0], &frame->uc.tuc_osf_sigmask);

    __put_user(target_sigaltstack_used.ss_sp,

               &frame->uc.tuc_stack.ss_sp);

    __put_user(sas_ss_flags(env->ir[IR_SP]),

               &frame->uc.tuc_stack.ss_flags);

    __put_user(target_sigaltstack_used.ss_size,

               &frame->uc.tuc_stack.ss_size);

    err |= setup_sigcontext(&frame->uc.tuc_mcontext, env, frame_addr, set);

    for (i = 0; i < TARGET_NSIG_WORDS; ++i) {

        __put_user(set->sig[i], &frame->uc.tuc_sigmask.sig[i]);

    }



    if (ka->sa_restorer) {

        r26 = ka->sa_restorer;

    } else {

        __put_user(INSN_MOV_R30_R16, &frame->retcode[0]);

        __put_user(INSN_LDI_R0 + TARGET_NR_rt_sigreturn,

                   &frame->retcode[1]);

        __put_user(INSN_CALLSYS, &frame->retcode[2]);

        /* imb(); */

        r26 = frame_addr;

    }



    if (err) {

    give_sigsegv:

       if (sig == TARGET_SIGSEGV) {

            ka->_sa_handler = TARGET_SIG_DFL;

        }

        force_sig(TARGET_SIGSEGV);

    }



    env->ir[IR_RA] = r26;

    env->ir[IR_PV] = env->pc = ka->_sa_handler;

    env->ir[IR_A0] = sig;

    env->ir[IR_A1] = frame_addr + offsetof(struct target_rt_sigframe, info);

    env->ir[IR_A2] = frame_addr + offsetof(struct target_rt_sigframe, uc);

    env->ir[IR_SP] = frame_addr;

}
