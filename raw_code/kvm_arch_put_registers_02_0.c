int kvm_arch_put_registers(CPUState *env, int level)

{

    int ret;



    assert(cpu_is_stopped(env) || qemu_cpu_self(env));



    ret = kvm_getput_regs(env, 1);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_xsave(env);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_xcrs(env);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_sregs(env);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_msrs(env, level);

    if (ret < 0) {

        return ret;

    }

    if (level >= KVM_PUT_RESET_STATE) {

        ret = kvm_put_mp_state(env);

        if (ret < 0) {

            return ret;

        }

    }

    ret = kvm_put_vcpu_events(env, level);

    if (ret < 0) {

        return ret;

    }

    ret = kvm_put_debugregs(env);

    if (ret < 0) {

        return ret;

    }

    /* must be last */

    ret = kvm_guest_debug_workarounds(env);

    if (ret < 0) {

        return ret;

    }

    return 0;

}
