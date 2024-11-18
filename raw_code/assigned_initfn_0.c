static int assigned_initfn(struct PCIDevice *pci_dev)

{

    AssignedDevice *dev = DO_UPCAST(AssignedDevice, dev, pci_dev);

    uint8_t e_intx;

    int r;

    Error *local_err = NULL;



    if (!kvm_enabled()) {

        error_report("pci-assign: error: requires KVM support");

        return -1;

    }



    if (!dev->host.domain && !dev->host.bus && !dev->host.slot &&

        !dev->host.function) {

        error_report("pci-assign: error: no host device specified");

        return -1;

    }



    /*

     * Set up basic config space access control. Will be further refined during

     * device initialization.

     */

    assigned_dev_emulate_config_read(dev, 0, PCI_CONFIG_SPACE_SIZE);

    assigned_dev_direct_config_read(dev, PCI_STATUS, 2);

    assigned_dev_direct_config_read(dev, PCI_REVISION_ID, 1);

    assigned_dev_direct_config_read(dev, PCI_CLASS_PROG, 3);

    assigned_dev_direct_config_read(dev, PCI_CACHE_LINE_SIZE, 1);

    assigned_dev_direct_config_read(dev, PCI_LATENCY_TIMER, 1);

    assigned_dev_direct_config_read(dev, PCI_BIST, 1);

    assigned_dev_direct_config_read(dev, PCI_CARDBUS_CIS, 4);

    assigned_dev_direct_config_read(dev, PCI_SUBSYSTEM_VENDOR_ID, 2);

    assigned_dev_direct_config_read(dev, PCI_SUBSYSTEM_ID, 2);

    assigned_dev_direct_config_read(dev, PCI_CAPABILITY_LIST + 1, 7);

    assigned_dev_direct_config_read(dev, PCI_MIN_GNT, 1);

    assigned_dev_direct_config_read(dev, PCI_MAX_LAT, 1);

    memcpy(dev->emulate_config_write, dev->emulate_config_read,

           sizeof(dev->emulate_config_read));



    get_real_device(dev, &local_err);

    if (local_err) {

        qerror_report_err(local_err);

        error_free(local_err);

        goto out;

    }



    if (assigned_device_pci_cap_init(pci_dev, &local_err) < 0) {

        qerror_report_err(local_err);

        error_free(local_err);

        goto out;

    }



    /* intercept MSI-X entry page in the MMIO */

    if (dev->cap.available & ASSIGNED_DEVICE_CAP_MSIX) {

        assigned_dev_register_msix_mmio(dev, &local_err);

        if (local_err) {

            qerror_report_err(local_err);

            error_free(local_err);

            goto out;

        }

    }



    /* handle real device's MMIO/PIO BARs */

    assigned_dev_register_regions(dev->real_device.regions,

                                  dev->real_device.region_number, dev,

                                  &local_err);

    if (local_err) {

        qerror_report_err(local_err);

        error_free(local_err);

        goto out;

    }



    /* handle interrupt routing */

    e_intx = dev->dev.config[PCI_INTERRUPT_PIN] - 1;

    dev->intpin = e_intx;

    dev->intx_route.mode = PCI_INTX_DISABLED;

    dev->intx_route.irq = -1;



    /* assign device to guest */

    r = assign_device(dev);

    if (r < 0) {

        goto out;

    }



    /* assign legacy INTx to the device */

    r = assign_intx(dev);

    if (r < 0) {

        goto assigned_out;

    }



    assigned_dev_load_option_rom(dev);



    add_boot_device_path(dev->bootindex, &pci_dev->qdev, NULL);



    return 0;



assigned_out:

    deassign_device(dev);

out:

    free_assigned_device(dev);

    return -1;

}
