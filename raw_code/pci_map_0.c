static void pci_map(PCIDevice * pci_dev, int region_num,

                    uint32_t addr, uint32_t size, int type)

{

    PCIEEPRO100State *d = DO_UPCAST(PCIEEPRO100State, dev, pci_dev);

    EEPRO100State *s = &d->eepro100;



    logout("region %d, addr=0x%08x, size=0x%08x, type=%d\n",

           region_num, addr, size, type);



    assert(region_num == 1);

    register_ioport_write(addr, size, 1, ioport_write1, s);

    register_ioport_read(addr, size, 1, ioport_read1, s);

    register_ioport_write(addr, size, 2, ioport_write2, s);

    register_ioport_read(addr, size, 2, ioport_read2, s);

    register_ioport_write(addr, size, 4, ioport_write4, s);

    register_ioport_read(addr, size, 4, ioport_read4, s);



    s->region[region_num] = addr;

}
