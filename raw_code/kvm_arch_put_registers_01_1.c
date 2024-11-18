int kvm_arch_put_registers(CPUState *env)

{

    struct kvm_regs regs;

    int ret;

    int i;



    ret = kvm_vcpu_ioctl(env, KVM_GET_REGS, &regs);

    if (ret < 0)

        return ret;



    regs.ctr = env->ctr;

    regs.lr  = env->lr;

    regs.xer = env->xer;

    regs.msr = env->msr;

    regs.pc = env->nip;



    regs.srr0 = env->spr[SPR_SRR0];

    regs.srr1 = env->spr[SPR_SRR1];



    regs.sprg0 = env->spr[SPR_SPRG0];

    regs.sprg1 = env->spr[SPR_SPRG1];

    regs.sprg2 = env->spr[SPR_SPRG2];

    regs.sprg3 = env->spr[SPR_SPRG3];

    regs.sprg4 = env->spr[SPR_SPRG4];

    regs.sprg5 = env->spr[SPR_SPRG5];

    regs.sprg6 = env->spr[SPR_SPRG6];

    regs.sprg7 = env->spr[SPR_SPRG7];



    for (i = 0;i < 32; i++)

        regs.gpr[i] = env->gpr[i];



    ret = kvm_vcpu_ioctl(env, KVM_SET_REGS, &regs);

    if (ret < 0)

        return ret;



    return ret;

}
