static int lsi_scsi_init(PCIDevice *dev)

{

    LSIState *s = DO_UPCAST(LSIState, dev, dev);

    uint8_t *pci_conf;



    pci_conf = s->dev.config;



    /* PCI Vendor ID (word) */

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_LSI_LOGIC);

    /* PCI device ID (word) */

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_LSI_53C895A);

    /* PCI base class code */

    pci_config_set_class(pci_conf, PCI_CLASS_STORAGE_SCSI);

    /* PCI subsystem ID */

    pci_conf[PCI_SUBSYSTEM_ID] = 0x00;

    pci_conf[PCI_SUBSYSTEM_ID + 1] = 0x10;

    /* PCI latency timer = 255 */

    pci_conf[PCI_LATENCY_TIMER] = 0xff;

    /* TODO: RST# value should be 0 */

    /* Interrupt pin 1 */

    pci_conf[PCI_INTERRUPT_PIN] = 0x01;



    s->mmio_io_addr = cpu_register_io_memory(lsi_mmio_readfn,

                                             lsi_mmio_writefn, s,

                                             DEVICE_NATIVE_ENDIAN);

    s->ram_io_addr = cpu_register_io_memory(lsi_ram_readfn,

                                            lsi_ram_writefn, s,

                                            DEVICE_NATIVE_ENDIAN);



    pci_register_bar(&s->dev, 0, 256,

                           PCI_BASE_ADDRESS_SPACE_IO, lsi_io_mapfunc);

    pci_register_bar_simple(&s->dev, 1, 0x400, 0, s->mmio_io_addr);

    pci_register_bar(&s->dev, 2, 0x2000,

                           PCI_BASE_ADDRESS_SPACE_MEMORY, lsi_ram_mapfunc);

    QTAILQ_INIT(&s->queue);



    scsi_bus_new(&s->bus, &dev->qdev, 1, LSI_MAX_DEVS, lsi_command_complete);

    if (!dev->qdev.hotplugged) {

        return scsi_bus_legacy_handle_cmdline(&s->bus);

    }

    return 0;

}
