static int kvm_get_fpu(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    struct kvm_fpu fpu;

    int i, ret;



    ret = kvm_vcpu_ioctl(CPU(cpu), KVM_GET_FPU, &fpu);

    if (ret < 0) {

        return ret;

    }



    env->fpstt = (fpu.fsw >> 11) & 7;

    env->fpus = fpu.fsw;

    env->fpuc = fpu.fcw;

    env->fpop = fpu.last_opcode;

    env->fpip = fpu.last_ip;

    env->fpdp = fpu.last_dp;

    for (i = 0; i < 8; ++i) {

        env->fptags[i] = !((fpu.ftwx >> i) & 1);

    }

    memcpy(env->fpregs, fpu.fpr, sizeof env->fpregs);

    memcpy(env->xmm_regs, fpu.xmm, sizeof env->xmm_regs);

    env->mxcsr = fpu.mxcsr;



    return 0;

}
