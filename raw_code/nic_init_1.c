static PCIDevice *nic_init(PCIBus * bus, NICInfo * nd, uint32_t device)

{

    PCIEEPRO100State *d;

    EEPRO100State *s;



    logout("\n");



    d = (PCIEEPRO100State *) pci_register_device(bus, nd->model,

                                                 sizeof(PCIEEPRO100State), -1,

                                                 NULL, NULL);



    s = &d->eepro100;

    s->device = device;

    s->pci_dev = &d->dev;



    pci_reset(s);



    /* Add 64 * 2 EEPROM. i82557 and i82558 support a 64 word EEPROM,

     * i82559 and later support 64 or 256 word EEPROM. */

    s->eeprom = eeprom93xx_new(EEPROM_SIZE);



    /* Handler for memory-mapped I/O */

    d->eepro100.mmio_index =

        cpu_register_io_memory(0, pci_mmio_read, pci_mmio_write, s);



    pci_register_io_region(&d->dev, 0, PCI_MEM_SIZE,

                           PCI_ADDRESS_SPACE_MEM |

                           PCI_ADDRESS_SPACE_MEM_PREFETCH, pci_mmio_map);

    pci_register_io_region(&d->dev, 1, PCI_IO_SIZE, PCI_ADDRESS_SPACE_IO,

                           pci_map);

    pci_register_io_region(&d->dev, 2, PCI_FLASH_SIZE, PCI_ADDRESS_SPACE_MEM,

                           pci_mmio_map);



    memcpy(s->macaddr, nd->macaddr, 6);

    logout("macaddr: %s\n", nic_dump(&s->macaddr[0], 6));

    assert(s->region[1] == 0);



    nic_reset(s);



    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 nic_receive, nic_can_receive, s);



    qemu_format_nic_info_str(s->vc, s->macaddr);



    qemu_register_reset(nic_reset, s);



    register_savevm(s->vc->model, -1, 3, nic_save, nic_load, s);

    return (PCIDevice *)d;

}
