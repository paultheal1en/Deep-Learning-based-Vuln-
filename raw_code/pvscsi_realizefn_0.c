pvscsi_realizefn(PCIDevice *pci_dev, Error **errp)

{

    PVSCSIState *s = PVSCSI(pci_dev);



    trace_pvscsi_state("init");



    /* PCI subsystem ID, subsystem vendor ID, revision */

    if (PVSCSI_USE_OLD_PCI_CONFIGURATION(s)) {

        pci_set_word(pci_dev->config + PCI_SUBSYSTEM_ID, 0x1000);

    } else {

        pci_set_word(pci_dev->config + PCI_SUBSYSTEM_VENDOR_ID,

                     PCI_VENDOR_ID_VMWARE);

        pci_set_word(pci_dev->config + PCI_SUBSYSTEM_ID,

                     PCI_DEVICE_ID_VMWARE_PVSCSI);

        pci_config_set_revision(pci_dev->config, 0x2);

    }



    /* PCI latency timer = 255 */

    pci_dev->config[PCI_LATENCY_TIMER] = 0xff;



    /* Interrupt pin A */

    pci_config_set_interrupt_pin(pci_dev->config, 1);



    memory_region_init_io(&s->io_space, OBJECT(s), &pvscsi_ops, s,

                          "pvscsi-io", PVSCSI_MEM_SPACE_SIZE);

    pci_register_bar(pci_dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->io_space);



    pvscsi_init_msi(s);



    if (pci_is_express(pci_dev) && pci_bus_is_express(pci_dev->bus)) {

        pcie_endpoint_cap_init(pci_dev, PVSCSI_EXP_EP_OFFSET);

    }



    s->completion_worker = qemu_bh_new(pvscsi_process_completion_queue, s);



    scsi_bus_new(&s->bus, sizeof(s->bus), DEVICE(pci_dev),

                 &pvscsi_scsi_info, NULL);

    /* override default SCSI bus hotplug-handler, with pvscsi's one */

    qbus_set_hotplug_handler(BUS(&s->bus), DEVICE(s), &error_abort);

    pvscsi_reset_state(s);

}
