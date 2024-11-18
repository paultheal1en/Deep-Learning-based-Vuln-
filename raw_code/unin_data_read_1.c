static uint64_t unin_data_read(void *opaque, hwaddr addr,

                               unsigned len)

{

    UNINState *s = opaque;

    PCIHostState *phb = PCI_HOST_BRIDGE(s);

    uint32_t val;



    val = pci_data_read(phb->bus,

                        unin_get_config_reg(phb->config_reg, addr),

                        len);

    UNIN_DPRINTF("read addr %" TARGET_FMT_plx " len %d val %x\n",

                 addr, len, val);

    return val;

}
