static void kvm_s390_flic_realize(DeviceState *dev, Error **errp)

{

    KVMS390FLICState *flic_state = KVM_S390_FLIC(dev);

    struct kvm_create_device cd = {0};

    struct kvm_device_attr test_attr = {0};

    int ret;

    Error *errp_local = NULL;







    flic_state->fd = -1;

    if (!kvm_check_extension(kvm_state, KVM_CAP_DEVICE_CTRL)) {

        error_setg_errno(&errp_local, errno, "KVM is missing capability"

                         " KVM_CAP_DEVICE_CTRL");

        trace_flic_no_device_api(errno);





    cd.type = KVM_DEV_TYPE_FLIC;

    ret = kvm_vm_ioctl(kvm_state, KVM_CREATE_DEVICE, &cd);

    if (ret < 0) {

        error_setg_errno(&errp_local, errno, "Creating the KVM device failed");

        trace_flic_create_device(errno);



    flic_state->fd = cd.fd;



    /* Check clear_io_irq support */

    test_attr.group = KVM_DEV_FLIC_CLEAR_IO_IRQ;

    flic_state->clear_io_supported = !ioctl(flic_state->fd,

                                            KVM_HAS_DEVICE_ATTR, test_attr);



    return;

fail:

    error_propagate(errp, errp_local);
