static int kvm_s390_register_io_adapter(S390FLICState *fs, uint32_t id,

                                        uint8_t isc, bool swap,

                                        bool is_maskable)

{

    struct kvm_s390_io_adapter adapter = {

        .id = id,

        .isc = isc,

        .maskable = is_maskable,

        .swap = swap,

    };

    KVMS390FLICState *flic = KVM_S390_FLIC(fs);

    int r, ret;

    struct kvm_device_attr attr = {

        .group = KVM_DEV_FLIC_ADAPTER_REGISTER,

        .addr = (uint64_t)&adapter,

    };



    if (!kvm_check_extension(kvm_state, KVM_CAP_IRQ_ROUTING)) {

        /* nothing to do */

        return 0;

    }



    r = ioctl(flic->fd, KVM_SET_DEVICE_ATTR, &attr);



    ret = r ? -errno : 0;

    return ret;

}
