static void setup_frame(int sig, struct target_sigaction *ka,

                        target_sigset_t *set, CPUAlphaState *env)

{

    abi_ulong frame_addr, r26;

    struct target_sigframe *frame;

    int err = 0;



    frame_addr = get_sigframe(ka, env, sizeof(*frame));

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0)) {

        goto give_sigsegv;

    }



    err |= setup_sigcontext(&frame->sc, env, frame_addr, set);



    if (ka->sa_restorer) {

        r26 = ka->sa_restorer;

    } else {

        __put_user(INSN_MOV_R30_R16, &frame->retcode[0]);

        __put_user(INSN_LDI_R0 + TARGET_NR_sigreturn,

                   &frame->retcode[1]);

        __put_user(INSN_CALLSYS, &frame->retcode[2]);

        /* imb() */

        r26 = frame_addr;

    }



    unlock_user_struct(frame, frame_addr, 1);



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

    env->ir[IR_A1] = 0;

    env->ir[IR_A2] = frame_addr + offsetof(struct target_sigframe, sc);

    env->ir[IR_SP] = frame_addr;

}
