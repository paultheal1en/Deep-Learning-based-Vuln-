static int kvm_put_msr_feature_control(X86CPU *cpu)

{

    struct {

        struct kvm_msrs info;

        struct kvm_msr_entry entry;

    } msr_data;



    kvm_msr_entry_set(&msr_data.entry, MSR_IA32_FEATURE_CONTROL,

                      cpu->env.msr_ia32_feature_control);

    msr_data.info.nmsrs = 1;

    return kvm_vcpu_ioctl(CPU(cpu), KVM_SET_MSRS, &msr_data);

}
