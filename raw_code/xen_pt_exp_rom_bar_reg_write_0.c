static int xen_pt_exp_rom_bar_reg_write(XenPCIPassthroughState *s,

                                        XenPTReg *cfg_entry, uint32_t *val,

                                        uint32_t dev_value, uint32_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    XenPTRegion *base = NULL;

    PCIDevice *d = (PCIDevice *)&s->dev;

    uint32_t writable_mask = 0;

    uint32_t throughable_mask = get_throughable_mask(s, reg, valid_mask);

    pcibus_t r_size = 0;

    uint32_t bar_ro_mask = 0;



    r_size = d->io_regions[PCI_ROM_SLOT].size;

    base = &s->bases[PCI_ROM_SLOT];

    /* align memory type resource size */

    r_size = xen_pt_get_emul_size(base->bar_flag, r_size);



    /* set emulate mask and read-only mask */

    bar_ro_mask = (reg->ro_mask | (r_size - 1)) & ~PCI_ROM_ADDRESS_ENABLE;



    /* modify emulate register */

    writable_mask = ~bar_ro_mask & valid_mask;

    cfg_entry->data = XEN_PT_MERGE_VALUE(*val, cfg_entry->data, writable_mask);



    /* create value for writing to I/O device register */

    *val = XEN_PT_MERGE_VALUE(*val, dev_value, throughable_mask);



    return 0;

}
