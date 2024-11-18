static int kvm_set_mce(CPUState *env, struct kvm_x86_mce *m)

{

    return kvm_vcpu_ioctl(env, KVM_X86_SET_MCE, m);

}
