static void spapr_powerdown_req(Notifier *n, void *opaque)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    struct rtas_error_log *hdr;

    struct rtas_event_log_v6 *v6hdr;

    struct rtas_event_log_v6_maina *maina;

    struct rtas_event_log_v6_mainb *mainb;

    struct rtas_event_log_v6_epow *epow;

    struct epow_log_full *new_epow;



    new_epow = g_malloc0(sizeof(*new_epow));

    hdr = &new_epow->hdr;

    v6hdr = &new_epow->v6hdr;

    maina = &new_epow->maina;

    mainb = &new_epow->mainb;

    epow = &new_epow->epow;



    hdr->summary = cpu_to_be32(RTAS_LOG_VERSION_6

                               | RTAS_LOG_SEVERITY_EVENT

                               | RTAS_LOG_DISPOSITION_NOT_RECOVERED

                               | RTAS_LOG_OPTIONAL_PART_PRESENT

                               | RTAS_LOG_TYPE_EPOW);

    hdr->extended_length = cpu_to_be32(sizeof(*new_epow)

                                       - sizeof(new_epow->hdr));



    spapr_init_v6hdr(v6hdr);

    spapr_init_maina(maina, 3 /* Main-A, Main-B and EPOW */);



    mainb->hdr.section_id = cpu_to_be16(RTAS_LOG_V6_SECTION_ID_MAINB);

    mainb->hdr.section_length = cpu_to_be16(sizeof(*mainb));

    /* FIXME: section version, subtype and creator id? */

    mainb->subsystem_id = 0xa0; /* External environment */

    mainb->event_severity = 0x00; /* Informational / non-error */

    mainb->event_subtype = 0xd0; /* Normal shutdown */



    epow->hdr.section_id = cpu_to_be16(RTAS_LOG_V6_SECTION_ID_EPOW);

    epow->hdr.section_length = cpu_to_be16(sizeof(*epow));

    epow->hdr.section_version = 2; /* includes extended modifier */

    /* FIXME: section subtype and creator id? */

    epow->sensor_value = RTAS_LOG_V6_EPOW_ACTION_SYSTEM_SHUTDOWN;

    epow->event_modifier = RTAS_LOG_V6_EPOW_MODIFIER_NORMAL;

    epow->extended_modifier = RTAS_LOG_V6_EPOW_XMODIFIER_PARTITION_SPECIFIC;



    rtas_event_log_queue(RTAS_LOG_TYPE_EPOW, new_epow);



    qemu_irq_pulse(xics_get_qirq(XICS_FABRIC(spapr),

                                 rtas_event_log_to_irq(spapr,

                                                       RTAS_LOG_TYPE_EPOW)));

}
