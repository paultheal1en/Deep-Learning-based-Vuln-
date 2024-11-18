static uint64_t apb_pci_config_read(void *opaque, target_phys_addr_t addr,

                                    unsigned size)

{

    uint32_t ret;

    APBState *s = opaque;



    ret = pci_data_read(s->bus, addr, size);

    ret = qemu_bswap_len(ret, size);

    APB_DPRINTF("%s: addr " TARGET_FMT_lx " -> %x\n", __func__, addr, ret);

    return ret;

}
