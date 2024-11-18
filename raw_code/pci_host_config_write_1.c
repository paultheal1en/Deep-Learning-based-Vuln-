static void pci_host_config_write(void *opaque, target_phys_addr_t addr,

                                  uint64_t val, unsigned len)

{

    PCIHostState *s = opaque;



    PCI_DPRINTF("%s addr " TARGET_FMT_plx " len %d val %"PRIx64"\n",

                __func__, addr, len, val);




    s->config_reg = val;
