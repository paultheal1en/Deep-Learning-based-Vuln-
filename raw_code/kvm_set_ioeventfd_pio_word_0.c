int kvm_set_ioeventfd_pio_word(int fd, uint16_t addr, uint16_t val, bool assign)

{

    struct kvm_ioeventfd kick = {

        .datamatch = val,

        .addr = addr,

        .len = 2,

        .flags = KVM_IOEVENTFD_FLAG_DATAMATCH | KVM_IOEVENTFD_FLAG_PIO,

        .fd = fd,

    };

    int r;

    if (!kvm_enabled())

        return -ENOSYS;

    if (!assign)

        kick.flags |= KVM_IOEVENTFD_FLAG_DEASSIGN;

    r = kvm_vm_ioctl(kvm_state, KVM_IOEVENTFD, &kick);

    if (r < 0)

        return r;

    return 0;

}
