static int kvm_getput_regs(CPUState *env, int set)

{

    struct kvm_regs regs;

    int ret = 0;



    if (!set) {

        ret = kvm_vcpu_ioctl(env, KVM_GET_REGS, &regs);

        if (ret < 0)

            return ret;

    }



    kvm_getput_reg(&regs.rax, &env->regs[R_EAX], set);

    kvm_getput_reg(&regs.rbx, &env->regs[R_EBX], set);

    kvm_getput_reg(&regs.rcx, &env->regs[R_ECX], set);

    kvm_getput_reg(&regs.rdx, &env->regs[R_EDX], set);

    kvm_getput_reg(&regs.rsi, &env->regs[R_ESI], set);

    kvm_getput_reg(&regs.rdi, &env->regs[R_EDI], set);

    kvm_getput_reg(&regs.rsp, &env->regs[R_ESP], set);

    kvm_getput_reg(&regs.rbp, &env->regs[R_EBP], set);

#ifdef TARGET_X86_64

    kvm_getput_reg(&regs.r8, &env->regs[8], set);

    kvm_getput_reg(&regs.r9, &env->regs[9], set);

    kvm_getput_reg(&regs.r10, &env->regs[10], set);

    kvm_getput_reg(&regs.r11, &env->regs[11], set);

    kvm_getput_reg(&regs.r12, &env->regs[12], set);

    kvm_getput_reg(&regs.r13, &env->regs[13], set);

    kvm_getput_reg(&regs.r14, &env->regs[14], set);

    kvm_getput_reg(&regs.r15, &env->regs[15], set);

#endif



    kvm_getput_reg(&regs.rflags, &env->eflags, set);

    kvm_getput_reg(&regs.rip, &env->eip, set);



    if (set)

        ret = kvm_vcpu_ioctl(env, KVM_SET_REGS, &regs);



    return ret;

}
