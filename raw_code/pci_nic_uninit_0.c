static int pci_nic_uninit(PCIDevice *dev)

{

    PCIEEPRO100State *d = DO_UPCAST(PCIEEPRO100State, dev, dev);

    EEPRO100State *s = &d->eepro100;



    cpu_unregister_io_memory(s->mmio_index);



    return 0;

}
