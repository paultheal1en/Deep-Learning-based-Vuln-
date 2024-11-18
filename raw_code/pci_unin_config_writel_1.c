static void pci_unin_config_writel (void *opaque, target_phys_addr_t addr,

                                    uint32_t val)

{

    UNINState *s = opaque;



    s->config_reg = val;

}
