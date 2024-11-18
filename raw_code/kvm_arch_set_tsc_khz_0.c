static int kvm_arch_set_tsc_khz(CPUState *cs)

{

    X86CPU *cpu = X86_CPU(cs);

    CPUX86State *env = &cpu->env;

    int r;



    if (!env->tsc_khz) {

        return 0;

    }



    r = kvm_check_extension(cs->kvm_state, KVM_CAP_TSC_CONTROL) ?

        kvm_vcpu_ioctl(cs, KVM_SET_TSC_KHZ, env->tsc_khz) :

        -ENOTSUP;

    if (r < 0) {

        /* When KVM_SET_TSC_KHZ fails, it's an error only if the current

         * TSC frequency doesn't match the one we want.

         */

        int cur_freq = kvm_check_extension(cs->kvm_state, KVM_CAP_GET_TSC_KHZ) ?

                       kvm_vcpu_ioctl(cs, KVM_GET_TSC_KHZ) :

                       -ENOTSUP;

        if (cur_freq <= 0 || cur_freq != env->tsc_khz) {

            error_report("warning: TSC frequency mismatch between "

                         "VM (%" PRId64 " kHz) and host (%d kHz), "

                         "and TSC scaling unavailable",

                         env->tsc_khz, cur_freq);

            return r;

        }

    }



    return 0;

}
