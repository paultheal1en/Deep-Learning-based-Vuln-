static int xen_pt_bar_reg_init(XenPCIPassthroughState *s, XenPTRegInfo *reg,

                               uint32_t real_offset, uint32_t *data)

{

    uint32_t reg_field = 0;

    int index;



    index = xen_pt_bar_offset_to_index(reg->offset);

    if (index < 0 || index >= PCI_NUM_REGIONS) {

        XEN_PT_ERR(&s->dev, "Internal error: Invalid BAR index [%d].\n", index);

        return -1;

    }



    /* set BAR flag */

    s->bases[index].bar_flag = xen_pt_bar_reg_parse(s, reg);

    if (s->bases[index].bar_flag == XEN_PT_BAR_FLAG_UNUSED) {

        reg_field = XEN_PT_INVALID_REG;

    }



    *data = reg_field;

    return 0;

}
