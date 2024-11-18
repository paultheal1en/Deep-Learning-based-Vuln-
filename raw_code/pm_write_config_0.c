static void pm_write_config(PCIDevice *d,

                            uint32_t address, uint32_t val, int len)

{

    DPRINTF("pm_write_config  address 0x%x  val 0x%x len 0x%x \n",

           address, val, len);

    pci_default_write_config(d, address, val, len);

}
