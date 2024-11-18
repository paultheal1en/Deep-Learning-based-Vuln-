static int kvm_s390_check_enable_cmma(KVMState *s)

{

    struct kvm_device_attr attr = {

        .group = KVM_S390_VM_MEM_CTRL,

        .attr = KVM_S390_VM_MEM_ENABLE_CMMA,

    };



    return kvm_vm_ioctl(s, KVM_HAS_DEVICE_ATTR, &attr);

}
