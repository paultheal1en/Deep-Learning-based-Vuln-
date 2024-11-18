int kvm_set_signal_mask(CPUState *env, const sigset_t *sigset)

{

    struct kvm_signal_mask *sigmask;

    int r;



    if (!sigset)

        return kvm_vcpu_ioctl(env, KVM_SET_SIGNAL_MASK, NULL);



    sigmask = qemu_malloc(sizeof(*sigmask) + sizeof(*sigset));



    sigmask->len = 8;

    memcpy(sigmask->sigset, sigset, sizeof(*sigset));

    r = kvm_vcpu_ioctl(env, KVM_SET_SIGNAL_MASK, sigmask);

    free(sigmask);



    return r;

}
