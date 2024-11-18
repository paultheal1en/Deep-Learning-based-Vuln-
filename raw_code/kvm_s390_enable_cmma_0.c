static void kvm_s390_enable_cmma(KVMState *s)

{

    int rc;

    struct kvm_device_attr attr = {

        .group = KVM_S390_VM_MEM_CTRL,

        .attr = KVM_S390_VM_MEM_ENABLE_CMMA,

    };



    if (kvm_s390_check_enable_cmma(s) || kvm_s390_check_clear_cmma(s)) {

        return;

    }



    rc = kvm_vm_ioctl(s, KVM_SET_DEVICE_ATTR, &attr);

    if (!rc) {

        qemu_register_reset(kvm_s390_clear_cmma_callback, s);

    }

    trace_kvm_enable_cmma(rc);

}
