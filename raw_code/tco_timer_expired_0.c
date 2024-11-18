static void tco_timer_expired(void *opaque)

{

    TCOIORegs *tr = opaque;

    ICH9LPCPMRegs *pm = container_of(tr, ICH9LPCPMRegs, tco_regs);

    ICH9LPCState *lpc = container_of(pm, ICH9LPCState, pm);

    uint32_t gcs = pci_get_long(lpc->chip_config + ICH9_CC_GCS);



    tr->tco.rld = 0;

    tr->tco.sts1 |= TCO_TIMEOUT;

    if (++tr->timeouts_no == 2) {

        tr->tco.sts2 |= TCO_SECOND_TO_STS;

        tr->tco.sts2 |= TCO_BOOT_STS;

        tr->timeouts_no = 0;



        if (!(gcs & ICH9_CC_GCS_NO_REBOOT)) {

            watchdog_perform_action();

            tco_timer_stop(tr);

            return;

        }

    }



    if (pm->smi_en & ICH9_PMIO_SMI_EN_TCO_EN) {

        ich9_generate_smi();

    } else {

        ich9_generate_nmi();

    }

    tr->tco.rld = tr->tco.tmr;

    tco_timer_reload(tr);

}
