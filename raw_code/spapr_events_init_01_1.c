void spapr_events_init(sPAPREnvironment *spapr)

{

    spapr->epow_irq = spapr_allocate_msi(0);

    spapr->epow_notifier.notify = spapr_powerdown_req;

    qemu_register_powerdown_notifier(&spapr->epow_notifier);

    spapr_rtas_register("check-exception", check_exception);

}
