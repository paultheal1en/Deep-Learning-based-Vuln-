static int xen_pt_status_reg_init(XenPCIPassthroughState *s,

                                  XenPTRegInfo *reg, uint32_t real_offset,

                                  uint32_t *data)

{

    XenPTRegGroup *reg_grp_entry = NULL;

    XenPTReg *reg_entry = NULL;

    uint32_t reg_field = 0;



    /* find Header register group */

    reg_grp_entry = xen_pt_find_reg_grp(s, PCI_CAPABILITY_LIST);

    if (reg_grp_entry) {

        /* find Capabilities Pointer register */

        reg_entry = xen_pt_find_reg(reg_grp_entry, PCI_CAPABILITY_LIST);

        if (reg_entry) {

            /* check Capabilities Pointer register */

            if (reg_entry->data) {

                reg_field |= PCI_STATUS_CAP_LIST;

            } else {

                reg_field &= ~PCI_STATUS_CAP_LIST;

            }

        } else {

            xen_shutdown_fatal_error("Internal error: Couldn't find XenPTReg*"

                                     " for Capabilities Pointer register."

                                     " (%s)\n", __func__);

            return -1;

        }

    } else {

        xen_shutdown_fatal_error("Internal error: Couldn't find XenPTRegGroup"

                                 " for Header. (%s)\n", __func__);

        return -1;

    }



    *data = reg_field;

    return 0;

}
