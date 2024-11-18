static void vmxnet3_pci_realize(PCIDevice *pci_dev, Error **errp)

{

    DeviceState *dev = DEVICE(pci_dev);

    VMXNET3State *s = VMXNET3(pci_dev);

    int ret;



    VMW_CBPRN("Starting init...");



    memory_region_init_io(&s->bar0, OBJECT(s), &b0_ops, s,

                          "vmxnet3-b0", VMXNET3_PT_REG_SIZE);

    pci_register_bar(pci_dev, VMXNET3_BAR0_IDX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &s->bar0);



    memory_region_init_io(&s->bar1, OBJECT(s), &b1_ops, s,

                          "vmxnet3-b1", VMXNET3_VD_REG_SIZE);

    pci_register_bar(pci_dev, VMXNET3_BAR1_IDX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &s->bar1);



    memory_region_init(&s->msix_bar, OBJECT(s), "vmxnet3-msix-bar",

                       VMXNET3_MSIX_BAR_SIZE);

    pci_register_bar(pci_dev, VMXNET3_MSIX_BAR_IDX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &s->msix_bar);



    vmxnet3_reset_interrupt_states(s);



    /* Interrupt pin A */

    pci_dev->config[PCI_INTERRUPT_PIN] = 0x01;



    ret = msi_init(pci_dev, VMXNET3_MSI_OFFSET(s), VMXNET3_MAX_NMSIX_INTRS,

                   VMXNET3_USE_64BIT, VMXNET3_PER_VECTOR_MASK, NULL);

    /* Any error other than -ENOTSUP(board's MSI support is broken)

     * is a programming error. Fall back to INTx silently on -ENOTSUP */

    assert(!ret || ret == -ENOTSUP);



    if (!vmxnet3_init_msix(s)) {

        VMW_WRPRN("Failed to initialize MSI-X, configuration is inconsistent.");

    }



    vmxnet3_net_init(s);



    if (pci_is_express(pci_dev)) {

        if (pci_bus_is_express(pci_dev->bus)) {

            pcie_endpoint_cap_init(pci_dev, VMXNET3_EXP_EP_OFFSET);

        }



        pcie_dev_ser_num_init(pci_dev, VMXNET3_DSN_OFFSET,

                              vmxnet3_device_serial_num(s));

    }



    register_savevm_live(dev, "vmxnet3-msix", -1, 1, &savevm_vmxnet3_msix, s);

}
