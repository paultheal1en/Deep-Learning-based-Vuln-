static void pci_unin_main_config_writel (void *opaque, target_phys_addr_t addr,

                                         uint32_t val)

{

    UNINState *s = opaque;



    UNIN_DPRINTF("config_writel addr " TARGET_FMT_plx " val %x\n", addr, val);

#ifdef TARGET_WORDS_BIGENDIAN

    val = bswap32(val);

#endif



    s->config_reg = val;

}
