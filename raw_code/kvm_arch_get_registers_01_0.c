int kvm_arch_get_registers(CPUState *cs)

{

    S390CPU *cpu = S390_CPU(cs);

    CPUS390XState *env = &cpu->env;

    struct kvm_one_reg reg;

    struct kvm_sregs sregs;

    struct kvm_regs regs;

    int i, r;



    /* get the PSW */

    env->psw.addr = cs->kvm_run->psw_addr;

    env->psw.mask = cs->kvm_run->psw_mask;



    /* the GPRS */

    if (cap_sync_regs && cs->kvm_run->kvm_valid_regs & KVM_SYNC_GPRS) {

        for (i = 0; i < 16; i++) {

            env->regs[i] = cs->kvm_run->s.regs.gprs[i];

        }

    } else {

        r = kvm_vcpu_ioctl(cs, KVM_GET_REGS, &regs);

        if (r < 0) {

            return r;

        }

         for (i = 0; i < 16; i++) {

            env->regs[i] = regs.gprs[i];

        }

    }



    /* The ACRS and CRS */

    if (cap_sync_regs &&

        cs->kvm_run->kvm_valid_regs & KVM_SYNC_ACRS &&

        cs->kvm_run->kvm_valid_regs & KVM_SYNC_CRS) {

        for (i = 0; i < 16; i++) {

            env->aregs[i] = cs->kvm_run->s.regs.acrs[i];

            env->cregs[i] = cs->kvm_run->s.regs.crs[i];

        }

    } else {

        r = kvm_vcpu_ioctl(cs, KVM_GET_SREGS, &sregs);

        if (r < 0) {

            return r;

        }

         for (i = 0; i < 16; i++) {

            env->aregs[i] = sregs.acrs[i];

            env->cregs[i] = sregs.crs[i];

        }

    }



    /* The prefix */

    if (cap_sync_regs && cs->kvm_run->kvm_valid_regs & KVM_SYNC_PREFIX) {

        env->psa = cs->kvm_run->s.regs.prefix;

    }



    /* One Regs */

    reg.id = KVM_REG_S390_CPU_TIMER;

    reg.addr = (__u64)&(env->cputm);

    r = kvm_vcpu_ioctl(cs, KVM_GET_ONE_REG, &reg);

    if (r < 0) {

        return r;

    }



    reg.id = KVM_REG_S390_CLOCK_COMP;

    reg.addr = (__u64)&(env->ckc);

    r = kvm_vcpu_ioctl(cs, KVM_GET_ONE_REG, &reg);

    if (r < 0) {

        return r;

    }



    reg.id = KVM_REG_S390_TODPR;

    reg.addr = (__u64)&(env->todpr);

    r = kvm_vcpu_ioctl(cs, KVM_GET_ONE_REG, &reg);

    if (r < 0) {

        return r;

    }



    if (cap_async_pf) {

        reg.id = KVM_REG_S390_PFTOKEN;

        reg.addr = (__u64)&(env->pfault_token);

        r = kvm_vcpu_ioctl(cs, KVM_GET_ONE_REG, &reg);

        if (r < 0) {

            return r;

        }



        reg.id = KVM_REG_S390_PFCOMPARE;

        reg.addr = (__u64)&(env->pfault_compare);

        r = kvm_vcpu_ioctl(cs, KVM_GET_ONE_REG, &reg);

        if (r < 0) {

            return r;

        }



        reg.id = KVM_REG_S390_PFSELECT;

        reg.addr = (__u64)&(env->pfault_select);

        r = kvm_vcpu_ioctl(cs, KVM_GET_ONE_REG, &reg);

        if (r < 0) {

            return r;

        }

    }



    return 0;

}
