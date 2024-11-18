static int setup_sigcontext(struct target_sigcontext *sc, CPUAlphaState *env,

                            abi_ulong frame_addr, target_sigset_t *set)

{

    int i, err = 0;



    __put_user(on_sig_stack(frame_addr), &sc->sc_onstack);

    __put_user(set->sig[0], &sc->sc_mask);

    __put_user(env->pc, &sc->sc_pc);

    __put_user(8, &sc->sc_ps);



    for (i = 0; i < 31; ++i) {

        __put_user(env->ir[i], &sc->sc_regs[i]);

    }

    __put_user(0, &sc->sc_regs[31]);



    for (i = 0; i < 31; ++i) {

        __put_user(env->fir[i], &sc->sc_fpregs[i]);

    }

    __put_user(0, &sc->sc_fpregs[31]);

    __put_user(cpu_alpha_load_fpcr(env), &sc->sc_fpcr);



    __put_user(0, &sc->sc_traparg_a0); /* FIXME */

    __put_user(0, &sc->sc_traparg_a1); /* FIXME */

    __put_user(0, &sc->sc_traparg_a2); /* FIXME */



    return err;

}
