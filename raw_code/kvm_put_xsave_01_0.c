static int kvm_put_xsave(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    struct kvm_xsave* xsave = env->kvm_xsave_buf;

    uint16_t cwd, swd, twd;

    int i, r;



    if (!kvm_has_xsave()) {

        return kvm_put_fpu(cpu);

    }



    memset(xsave, 0, sizeof(struct kvm_xsave));

    twd = 0;

    swd = env->fpus & ~(7 << 11);

    swd |= (env->fpstt & 7) << 11;

    cwd = env->fpuc;

    for (i = 0; i < 8; ++i) {

        twd |= (!env->fptags[i]) << i;

    }

    xsave->region[XSAVE_FCW_FSW] = (uint32_t)(swd << 16) + cwd;

    xsave->region[XSAVE_FTW_FOP] = (uint32_t)(env->fpop << 16) + twd;

    memcpy(&xsave->region[XSAVE_CWD_RIP], &env->fpip, sizeof(env->fpip));

    memcpy(&xsave->region[XSAVE_CWD_RDP], &env->fpdp, sizeof(env->fpdp));

    memcpy(&xsave->region[XSAVE_ST_SPACE], env->fpregs,

            sizeof env->fpregs);

    memcpy(&xsave->region[XSAVE_XMM_SPACE], env->xmm_regs,

            sizeof env->xmm_regs);

    xsave->region[XSAVE_MXCSR] = env->mxcsr;

    *(uint64_t *)&xsave->region[XSAVE_XSTATE_BV] = env->xstate_bv;

    memcpy(&xsave->region[XSAVE_YMMH_SPACE], env->ymmh_regs,

            sizeof env->ymmh_regs);

    memcpy(&xsave->region[XSAVE_BNDREGS], env->bnd_regs,

            sizeof env->bnd_regs);

    memcpy(&xsave->region[XSAVE_BNDCSR], &env->bndcs_regs,

            sizeof(env->bndcs_regs));

    memcpy(&xsave->region[XSAVE_OPMASK], env->opmask_regs,

            sizeof env->opmask_regs);

    memcpy(&xsave->region[XSAVE_ZMM_Hi256], env->zmmh_regs,

            sizeof env->zmmh_regs);

#ifdef TARGET_X86_64

    memcpy(&xsave->region[XSAVE_Hi16_ZMM], env->hi16_zmm_regs,

            sizeof env->hi16_zmm_regs);

#endif

    r = kvm_vcpu_ioctl(CPU(cpu), KVM_SET_XSAVE, xsave);

    return r;

}
