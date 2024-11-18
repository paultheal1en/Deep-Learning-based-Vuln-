void *lsi_scsi_init(PCIBus *bus, int devfn)

{

    LSIState *s;



    s = (LSIState *)pci_register_device(bus, "LSI53C895A SCSI HBA",

                                        sizeof(*s), devfn, NULL, NULL);

    if (s == NULL) {

        fprintf(stderr, "lsi-scsi: Failed to register PCI device\n");

        return NULL;

    }



    s->pci_dev.config[0x00] = 0x00;

    s->pci_dev.config[0x01] = 0x10;

    s->pci_dev.config[0x02] = 0x12;

    s->pci_dev.config[0x03] = 0x00;

    s->pci_dev.config[0x0b] = 0x01;

    s->pci_dev.config[0x3d] = 0x01; /* interrupt pin 1 */



    s->mmio_io_addr = cpu_register_io_memory(0, lsi_mmio_readfn,

                                             lsi_mmio_writefn, s);

    s->ram_io_addr = cpu_register_io_memory(0, lsi_ram_readfn,

                                            lsi_ram_writefn, s);



    pci_register_io_region((struct PCIDevice *)s, 0, 256,

                           PCI_ADDRESS_SPACE_IO, lsi_io_mapfunc);

    pci_register_io_region((struct PCIDevice *)s, 1, 0x400,

                           PCI_ADDRESS_SPACE_MEM, lsi_mmio_mapfunc);

    pci_register_io_region((struct PCIDevice *)s, 2, 0x2000,

                           PCI_ADDRESS_SPACE_MEM, lsi_ram_mapfunc);

    s->queue = qemu_malloc(sizeof(lsi_queue));

    s->queue_len = 1;

    s->active_commands = 0;



    lsi_soft_reset(s);



    return s;

}
