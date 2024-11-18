static int kvm_put_tscdeadline_msr(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    struct {

        struct kvm_msrs info;

        struct kvm_msr_entry entries[1];

    } msr_data;

    struct kvm_msr_entry *msrs = msr_data.entries;



    if (!has_msr_tsc_deadline) {

        return 0;

    }



    kvm_msr_entry_set(&msrs[0], MSR_IA32_TSCDEADLINE, env->tsc_deadline);



    msr_data.info.nmsrs = 1;



    return kvm_vcpu_ioctl(CPU(cpu), KVM_SET_MSRS, &msr_data);

}
