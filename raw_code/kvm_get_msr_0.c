static int kvm_get_msr(CPUState *env, struct kvm_msr_entry *msrs, int n)

{

    struct kvm_msrs *kmsrs = qemu_malloc(sizeof *kmsrs + n * sizeof *msrs);

    int r;



    kmsrs->nmsrs = n;

    memcpy(kmsrs->entries, msrs, n * sizeof *msrs);

    r = kvm_vcpu_ioctl(env, KVM_GET_MSRS, kmsrs);

    memcpy(msrs, kmsrs->entries, n * sizeof *msrs);

    free(kmsrs);

    return r;

}
