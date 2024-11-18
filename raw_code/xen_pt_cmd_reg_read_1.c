static int xen_pt_cmd_reg_read(XenPCIPassthroughState *s, XenPTReg *cfg_entry,

                               uint16_t *value, uint16_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    uint16_t valid_emu_mask = 0;

    uint16_t emu_mask = reg->emu_mask;



    if (s->is_virtfn) {

        emu_mask |= PCI_COMMAND_MEMORY;

    }



    /* emulate word register */

    valid_emu_mask = emu_mask & valid_mask;

    *value = XEN_PT_MERGE_VALUE(*value, cfg_entry->data, ~valid_emu_mask);



    return 0;

}
