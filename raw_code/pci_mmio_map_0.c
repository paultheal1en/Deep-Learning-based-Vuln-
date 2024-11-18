static void pci_mmio_map(PCIDevice * pci_dev, int region_num,

                         uint32_t addr, uint32_t size, int type)

{

    PCIEEPRO100State *d = DO_UPCAST(PCIEEPRO100State, dev, pci_dev);



    logout("region %d, addr=0x%08x, size=0x%08x, type=%d\n",

           region_num, addr, size, type);



    if (region_num == 0) {

        /* Map control / status registers. */

        cpu_register_physical_memory(addr, size, d->eepro100.mmio_index);

        d->eepro100.region[region_num] = addr;

    }

}
