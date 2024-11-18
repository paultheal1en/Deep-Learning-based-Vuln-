int kvmppc_get_htab_fd(bool write)

{

    struct kvm_get_htab_fd s = {

        .flags = write ? KVM_GET_HTAB_WRITE : 0,

        .start_index = 0,

    };



    if (!cap_htab_fd) {

        fprintf(stderr, "KVM version doesn't support saving the hash table\n");

        return -1;

    }



    return kvm_vm_ioctl(kvm_state, KVM_PPC_GET_HTAB_FD, &s);

}
