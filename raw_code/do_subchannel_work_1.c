static int do_subchannel_work(SubchDev *sch)

{

    if (!sch->do_subchannel_work) {

        return -EINVAL;

    }

    g_assert(sch->curr_status.scsw.ctrl & SCSW_CTRL_MASK_FCTL);

    return sch->do_subchannel_work(sch);

}
