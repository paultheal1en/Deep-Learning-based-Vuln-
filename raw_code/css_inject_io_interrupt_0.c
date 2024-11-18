static void css_inject_io_interrupt(SubchDev *sch)

{

    S390CPU *cpu = s390_cpu_addr2state(0);

    uint8_t isc = (sch->curr_status.pmcw.flags & PMCW_FLAGS_MASK_ISC) >> 11;



    trace_css_io_interrupt(sch->cssid, sch->ssid, sch->schid,

                           sch->curr_status.pmcw.intparm, isc, "");

    s390_io_interrupt(cpu,

                      css_build_subchannel_id(sch),

                      sch->schid,

                      sch->curr_status.pmcw.intparm,

                      isc << 27);

}
