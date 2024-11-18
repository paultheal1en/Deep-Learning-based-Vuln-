static int target_restore_sigframe(CPUARMState *env,

                                   struct target_rt_sigframe *sf)

{

    sigset_t set;

    int i;

    struct target_aux_context *aux =

        (struct target_aux_context *)sf->uc.tuc_mcontext.__reserved;

    uint32_t magic, size, fpsr, fpcr;

    uint64_t pstate;



    target_to_host_sigset(&set, &sf->uc.tuc_sigmask);

    sigprocmask(SIG_SETMASK, &set, NULL);



    for (i = 0; i < 31; i++) {

        __get_user(env->xregs[i], &sf->uc.tuc_mcontext.regs[i]);

    }



    __get_user(env->xregs[31], &sf->uc.tuc_mcontext.sp);

    __get_user(env->pc, &sf->uc.tuc_mcontext.pc);

    __get_user(pstate, &sf->uc.tuc_mcontext.pstate);

    pstate_write(env, pstate);



    __get_user(magic, &aux->fpsimd.head.magic);

    __get_user(size, &aux->fpsimd.head.size);



    if (magic != TARGET_FPSIMD_MAGIC

        || size != sizeof(struct target_fpsimd_context)) {

        return 1;

    }



    for (i = 0; i < 32; i++) {

#ifdef TARGET_WORDS_BIGENDIAN

        __get_user(env->vfp.regs[i * 2], &aux->fpsimd.vregs[i * 2 + 1]);

        __get_user(env->vfp.regs[i * 2 + 1], &aux->fpsimd.vregs[i * 2]);

#else

        __get_user(env->vfp.regs[i * 2], &aux->fpsimd.vregs[i * 2]);

        __get_user(env->vfp.regs[i * 2 + 1], &aux->fpsimd.vregs[i * 2 + 1]);

#endif

    }

    __get_user(fpsr, &aux->fpsimd.fpsr);

    vfp_set_fpsr(env, fpsr);

    __get_user(fpcr, &aux->fpsimd.fpcr);

    vfp_set_fpcr(env, fpcr);



    return 0;

}
