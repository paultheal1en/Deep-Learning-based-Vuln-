static void xics_kvm_realize(DeviceState *dev, Error **errp)

{

    KVMXICSState *icpkvm = KVM_XICS(dev);

    XICSState *icp = XICS_COMMON(dev);

    int i, rc;

    Error *error = NULL;

    struct kvm_create_device xics_create_device = {

        .type = KVM_DEV_TYPE_XICS,

        .flags = 0,

    };



    if (!kvm_enabled() || !kvm_check_extension(kvm_state, KVM_CAP_IRQ_XICS)) {

        error_setg(errp,

                   "KVM and IRQ_XICS capability must be present for in-kernel XICS");

        goto fail;

    }



    icpkvm->set_xive_token = spapr_rtas_register("ibm,set-xive", rtas_dummy);

    icpkvm->get_xive_token = spapr_rtas_register("ibm,get-xive", rtas_dummy);

    icpkvm->int_off_token = spapr_rtas_register("ibm,int-off", rtas_dummy);

    icpkvm->int_on_token = spapr_rtas_register("ibm,int-on", rtas_dummy);



    rc = kvmppc_define_rtas_kernel_token(icpkvm->set_xive_token,

                                         "ibm,set-xive");

    if (rc < 0) {

        error_setg(errp, "kvmppc_define_rtas_kernel_token: ibm,set-xive");

        goto fail;

    }



    rc = kvmppc_define_rtas_kernel_token(icpkvm->get_xive_token,

                                         "ibm,get-xive");

    if (rc < 0) {

        error_setg(errp, "kvmppc_define_rtas_kernel_token: ibm,get-xive");

        goto fail;

    }



    rc = kvmppc_define_rtas_kernel_token(icpkvm->int_on_token, "ibm,int-on");

    if (rc < 0) {

        error_setg(errp, "kvmppc_define_rtas_kernel_token: ibm,int-on");

        goto fail;

    }



    rc = kvmppc_define_rtas_kernel_token(icpkvm->int_off_token, "ibm,int-off");

    if (rc < 0) {

        error_setg(errp, "kvmppc_define_rtas_kernel_token: ibm,int-off");

        goto fail;

    }



    /* Create the kernel ICP */

    rc = kvm_vm_ioctl(kvm_state, KVM_CREATE_DEVICE, &xics_create_device);

    if (rc < 0) {

        error_setg_errno(errp, -rc, "Error on KVM_CREATE_DEVICE for XICS");

        goto fail;

    }



    icpkvm->kernel_xics_fd = xics_create_device.fd;



    object_property_set_bool(OBJECT(icp->ics), true, "realized", &error);

    if (error) {

        error_propagate(errp, error);

        goto fail;

    }



    assert(icp->nr_servers);

    for (i = 0; i < icp->nr_servers; i++) {

        object_property_set_bool(OBJECT(&icp->ss[i]), true, "realized", &error);

        if (error) {

            error_propagate(errp, error);

            goto fail;

        }

    }



    kvm_kernel_irqchip = true;

    kvm_irqfds_allowed = true;

    kvm_msi_via_irqfd_allowed = true;

    kvm_gsi_direct_mapping = true;



    return;



fail:

    kvmppc_define_rtas_kernel_token(0, "ibm,set-xive");

    kvmppc_define_rtas_kernel_token(0, "ibm,get-xive");

    kvmppc_define_rtas_kernel_token(0, "ibm,int-on");

    kvmppc_define_rtas_kernel_token(0, "ibm,int-off");

}
