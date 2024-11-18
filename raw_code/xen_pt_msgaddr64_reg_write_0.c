static int xen_pt_msgaddr64_reg_write(XenPCIPassthroughState *s,

                                      XenPTReg *cfg_entry, uint32_t *val,

                                      uint32_t dev_value, uint32_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    uint32_t writable_mask = 0;

    uint32_t old_addr = cfg_entry->data;



    /* check whether the type is 64 bit or not */

    if (!(s->msi->flags & PCI_MSI_FLAGS_64BIT)) {

        XEN_PT_ERR(&s->dev,

                   "Can't write to the upper address without 64 bit support\n");

        return -1;

    }



    /* modify emulate register */

    writable_mask = reg->emu_mask & ~reg->ro_mask & valid_mask;

    cfg_entry->data = XEN_PT_MERGE_VALUE(*val, cfg_entry->data, writable_mask);

    /* update the msi_info too */

    s->msi->addr_hi = cfg_entry->data;



    /* create value for writing to I/O device register */

    *val = XEN_PT_MERGE_VALUE(*val, dev_value, 0);



    /* update MSI */

    if (cfg_entry->data != old_addr) {

        if (s->msi->mapped) {

            xen_pt_msi_update(s);

        }

    }



    return 0;

}
