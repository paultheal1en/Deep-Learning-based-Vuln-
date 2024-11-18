int do_subchannel_work_passthrough(SubchDev *sch)

{

    int ret = 0;

    SCSW *s = &sch->curr_status.scsw;



    if (s->ctrl & SCSW_FCTL_CLEAR_FUNC) {

        /* TODO: Clear handling */

        sch_handle_clear_func(sch);

    } else if (s->ctrl & SCSW_FCTL_HALT_FUNC) {

        /* TODO: Halt handling */

        sch_handle_halt_func(sch);

    } else if (s->ctrl & SCSW_FCTL_START_FUNC) {

        ret = sch_handle_start_func_passthrough(sch);

    }



    return ret;

}
