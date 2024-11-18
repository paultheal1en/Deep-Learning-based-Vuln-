static int assign_device(AssignedDevice *dev)

{

    uint32_t flags = KVM_DEV_ASSIGN_ENABLE_IOMMU;

    int r;



    /* Only pass non-zero PCI segment to capable module */

    if (!kvm_check_extension(kvm_state, KVM_CAP_PCI_SEGMENT) &&

        dev->host.domain) {

        error_report("Can't assign device inside non-zero PCI segment "

                     "as this KVM module doesn't support it.");

        return -ENODEV;

    }



    if (!kvm_check_extension(kvm_state, KVM_CAP_IOMMU)) {

        error_report("No IOMMU found.  Unable to assign device \"%s\"",

                     dev->dev.qdev.id);

        return -ENODEV;

    }



    if (dev->features & ASSIGNED_DEVICE_SHARE_INTX_MASK &&

        kvm_has_intx_set_mask()) {

        flags |= KVM_DEV_ASSIGN_PCI_2_3;

    }



    r = kvm_device_pci_assign(kvm_state, &dev->host, flags, &dev->dev_id);

    if (r < 0) {

        switch (r) {

        case -EBUSY: {

            char *cause;



            cause = assign_failed_examine(dev);

            error_report("Failed to assign device \"%s\" : %s\n%s",

                         dev->dev.qdev.id, strerror(-r), cause);

            g_free(cause);

            break;

        }

        default:

            error_report("Failed to assign device \"%s\" : %s",

                         dev->dev.qdev.id, strerror(-r));

            break;

        }

    }

    return r;

}
