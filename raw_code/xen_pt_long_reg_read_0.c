static int xen_pt_long_reg_read(XenPCIPassthroughState *s, XenPTReg *cfg_entry,

                                uint32_t *value, uint32_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    uint32_t valid_emu_mask = 0;



    /* emulate long register */

    valid_emu_mask = reg->emu_mask & valid_mask;

    *value = XEN_PT_MERGE_VALUE(*value, cfg_entry->data, ~valid_emu_mask);



    return 0;

}
