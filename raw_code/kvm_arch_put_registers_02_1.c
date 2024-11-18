int kvm_arch_put_registers(CPUState *cs, int level)

{

    struct kvm_one_reg reg;

    uint32_t fpr;

    uint64_t val;

    int i;

    int ret;



    ARMCPU *cpu = ARM_CPU(cs);

    CPUARMState *env = &cpu->env;



    /* If we are in AArch32 mode then we need to copy the AArch32 regs to the

     * AArch64 registers before pushing them out to 64-bit KVM.

     */

    if (!is_a64(env)) {

        aarch64_sync_32_to_64(env);

    }



    for (i = 0; i < 31; i++) {

        reg.id = AARCH64_CORE_REG(regs.regs[i]);

        reg.addr = (uintptr_t) &env->xregs[i];

        ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

        if (ret) {

            return ret;

        }

    }



    /* KVM puts SP_EL0 in regs.sp and SP_EL1 in regs.sp_el1. On the

     * QEMU side we keep the current SP in xregs[31] as well.

     */

    aarch64_save_sp(env, 1);



    reg.id = AARCH64_CORE_REG(regs.sp);

    reg.addr = (uintptr_t) &env->sp_el[0];

    ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

    if (ret) {

        return ret;

    }



    reg.id = AARCH64_CORE_REG(sp_el1);

    reg.addr = (uintptr_t) &env->sp_el[1];

    ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

    if (ret) {

        return ret;

    }



    /* Note that KVM thinks pstate is 64 bit but we use a uint32_t */

    if (is_a64(env)) {

        val = pstate_read(env);

    } else {

        val = cpsr_read(env);

    }

    reg.id = AARCH64_CORE_REG(regs.pstate);

    reg.addr = (uintptr_t) &val;

    ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

    if (ret) {

        return ret;

    }



    reg.id = AARCH64_CORE_REG(regs.pc);

    reg.addr = (uintptr_t) &env->pc;

    ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

    if (ret) {

        return ret;

    }



    reg.id = AARCH64_CORE_REG(elr_el1);

    reg.addr = (uintptr_t) &env->elr_el[1];

    ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

    if (ret) {

        return ret;

    }



    for (i = 0; i < KVM_NR_SPSR; i++) {

        reg.id = AARCH64_CORE_REG(spsr[i]);

        reg.addr = (uintptr_t) &env->banked_spsr[i - 1];

        ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

        if (ret) {

            return ret;

        }

    }



    /* Advanced SIMD and FP registers

     * We map Qn = regs[2n+1]:regs[2n]

     */

    for (i = 0; i < 32; i++) {

        int rd = i << 1;

        uint64_t fp_val[2];

#ifdef HOST_WORDS_BIGENDIAN

        fp_val[0] = env->vfp.regs[rd + 1];

        fp_val[1] = env->vfp.regs[rd];

#else

        fp_val[1] = env->vfp.regs[rd + 1];

        fp_val[0] = env->vfp.regs[rd];

#endif

        reg.id = AARCH64_SIMD_CORE_REG(fp_regs.vregs[i]);

        reg.addr = (uintptr_t)(&fp_val);

        ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

        if (ret) {

            return ret;

        }

    }



    reg.addr = (uintptr_t)(&fpr);

    fpr = vfp_get_fpsr(env);

    reg.id = AARCH64_SIMD_CTRL_REG(fp_regs.fpsr);

    ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

    if (ret) {

        return ret;

    }



    fpr = vfp_get_fpcr(env);

    reg.id = AARCH64_SIMD_CTRL_REG(fp_regs.fpcr);

    ret = kvm_vcpu_ioctl(cs, KVM_SET_ONE_REG, &reg);

    if (ret) {

        return ret;

    }



    if (!write_list_to_kvmstate(cpu)) {

        return EINVAL;

    }



    kvm_arm_sync_mpstate_to_kvm(cpu);



    return ret;

}
