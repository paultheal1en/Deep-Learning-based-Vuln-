long do_sigreturn(CPUAlphaState *env)

{

    struct target_sigcontext *sc;

    abi_ulong sc_addr = env->ir[IR_A0];

    target_sigset_t target_set;

    sigset_t set;



    if (!lock_user_struct(VERIFY_READ, sc, sc_addr, 1)) {

        goto badframe;

    }



    target_sigemptyset(&target_set);

    if (__get_user(target_set.sig[0], &sc->sc_mask)) {

        goto badframe;

    }



    target_to_host_sigset_internal(&set, &target_set);

    do_sigprocmask(SIG_SETMASK, &set, NULL);



    if (restore_sigcontext(env, sc)) {

        goto badframe;

    }

    unlock_user_struct(sc, sc_addr, 0);

    return env->ir[IR_V0];



 badframe:

    unlock_user_struct(sc, sc_addr, 0);

    force_sig(TARGET_SIGSEGV);

}
