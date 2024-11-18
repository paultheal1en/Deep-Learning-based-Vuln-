int kvm_arch_put_registers(CPUState *cpu, int level)

{

    X86CPU *x86_cpu = X86_CPU(cpu);

    int ret;



    assert(cpu_is_stopped(cpu) || qemu_cpu_is_self(cpu));



    if (level >= KVM_PUT_RESET_STATE) {

        ret = kvm_put_msr_feature_control(x86_cpu);

        if (ret < 0) {

            return ret;

        }

    }



    if (level == KVM_PUT_FULL_STATE) {

        /* We don't check for kvm_arch_set_tsc_khz() errors here,

         * because TSC frequency mismatch shouldn't abort migration,

         * unless the user explicitly asked for a more strict TSC

         * setting (e.g. using an explicit "tsc-freq" option).

         */

        kvm_arch_set_tsc_khz(cpu);

    }



    ret = kvm_getput_regs(x86_cpu, 1);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_xsave(x86_cpu);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_xcrs(x86_cpu);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_sregs(x86_cpu);

    if (ret < 0) {

        return ret;

    }

    /* must be before kvm_put_msrs */

    ret = kvm_inject_mce_oldstyle(x86_cpu);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_msrs(x86_cpu, level);

    if (ret < 0) {

        return ret;

    }

    if (level >= KVM_PUT_RESET_STATE) {

        ret = kvm_put_mp_state(x86_cpu);

        if (ret < 0) {

            return ret;

        }

    }



    ret = kvm_put_tscdeadline_msr(x86_cpu);

    if (ret < 0) {

        return ret;

    }



    ret = kvm_put_vcpu_events(x86_cpu, level);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_debugregs(x86_cpu);

    if (ret < 0) {

        return ret;

    }

    /* must be last */

    ret = kvm_guest_debug_workarounds(x86_cpu);

    if (ret < 0) {

        return ret;

    }

    return 0;

}
