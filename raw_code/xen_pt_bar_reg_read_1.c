static int xen_pt_bar_reg_read(XenPCIPassthroughState *s, XenPTReg *cfg_entry,

                               uint32_t *value, uint32_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    uint32_t valid_emu_mask = 0;

    uint32_t bar_emu_mask = 0;

    int index;



    /* get BAR index */

    index = xen_pt_bar_offset_to_index(reg->offset);

    if (index < 0 || index >= PCI_NUM_REGIONS) {

        XEN_PT_ERR(&s->dev, "Internal error: Invalid BAR index [%d].\n", index);

        return -1;

    }



    /* use fixed-up value from kernel sysfs */

    *value = base_address_with_flags(&s->real_device.io_regions[index]);



    /* set emulate mask depend on BAR flag */

    switch (s->bases[index].bar_flag) {

    case XEN_PT_BAR_FLAG_MEM:

        bar_emu_mask = XEN_PT_BAR_MEM_EMU_MASK;

        break;

    case XEN_PT_BAR_FLAG_IO:

        bar_emu_mask = XEN_PT_BAR_IO_EMU_MASK;

        break;

    case XEN_PT_BAR_FLAG_UPPER:

        bar_emu_mask = XEN_PT_BAR_ALLF;

        break;

    default:

        break;

    }



    /* emulate BAR */

    valid_emu_mask = bar_emu_mask & valid_mask;

    *value = XEN_PT_MERGE_VALUE(*value, cfg_entry->data, ~valid_emu_mask);



    return 0;

}
