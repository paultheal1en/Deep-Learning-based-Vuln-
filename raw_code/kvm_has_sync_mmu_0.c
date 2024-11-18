int kvm_has_sync_mmu(void)

{

    return kvm_check_extension(kvm_state, KVM_CAP_SYNC_MMU);

}
