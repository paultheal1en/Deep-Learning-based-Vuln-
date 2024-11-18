void css_conditional_io_interrupt(SubchDev *sch)

{

    /*

     * If the subchannel is not currently status pending, make it pending

     * with alert status.

     */

    if (!(sch->curr_status.scsw.ctrl & SCSW_STCTL_STATUS_PEND)) {

        S390CPU *cpu = s390_cpu_addr2state(0);

        uint8_t isc = (sch->curr_status.pmcw.flags & PMCW_FLAGS_MASK_ISC) >> 11;



        trace_css_io_interrupt(sch->cssid, sch->ssid, sch->schid,

                               sch->curr_status.pmcw.intparm, isc,

                               "(unsolicited)");

        sch->curr_status.scsw.ctrl &= ~SCSW_CTRL_MASK_STCTL;

        sch->curr_status.scsw.ctrl |=

            SCSW_STCTL_ALERT | SCSW_STCTL_STATUS_PEND;

        /* Inject an I/O interrupt. */

        s390_io_interrupt(cpu,

                          css_build_subchannel_id(sch),

                          sch->schid,

                          sch->curr_status.pmcw.intparm,

                          (0x80 >> isc) << 24);

    }

}
