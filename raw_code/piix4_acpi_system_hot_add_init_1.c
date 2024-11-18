static void piix4_acpi_system_hot_add_init(PCIBus *bus, PIIX4PMState *s)

{



    register_ioport_write(GPE_BASE, GPE_LEN, 1, gpe_writeb, s);

    register_ioport_read(GPE_BASE, GPE_LEN, 1,  gpe_readb, s);

    acpi_gpe_blk(&s->ar, GPE_BASE);



    register_ioport_read(PCI_UP_BASE, 4, 4, pci_up_read, s);

    register_ioport_read(PCI_DOWN_BASE, 4, 4, pci_down_read, s);



    register_ioport_write(PCI_EJ_BASE, 4, 4, pciej_write, bus);

    register_ioport_read(PCI_EJ_BASE, 4, 4,  pciej_read, bus);



    register_ioport_write(PCI_RMV_BASE, 4, 4, pcirmv_write, s);

    register_ioport_read(PCI_RMV_BASE, 4, 4,  pcirmv_read, s);



    pci_bus_hotplug(bus, piix4_device_hotplug, &s->dev.qdev);

}
