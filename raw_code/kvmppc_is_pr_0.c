static bool kvmppc_is_pr(KVMState *ks)

{

    /* Assume KVM-PR if the GET_PVINFO capability is available */

    return kvm_check_extension(ks, KVM_CAP_PPC_GET_PVINFO) != 0;

}
