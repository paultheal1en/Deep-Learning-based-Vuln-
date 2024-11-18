static void intel_hda_corb_run(IntelHDAState *d)

{

    hwaddr addr;

    uint32_t rp, verb;



    if (d->ics & ICH6_IRS_BUSY) {

        dprint(d, 2, "%s: [icw] verb 0x%08x\n", __FUNCTION__, d->icw);

        intel_hda_send_command(d, d->icw);

        return;

    }



    for (;;) {

        if (!(d->corb_ctl & ICH6_CORBCTL_RUN)) {

            dprint(d, 2, "%s: !run\n", __FUNCTION__);

            return;

        }

        if ((d->corb_rp & 0xff) == d->corb_wp) {

            dprint(d, 2, "%s: corb ring empty\n", __FUNCTION__);

            return;

        }

        if (d->rirb_count == d->rirb_cnt) {

            dprint(d, 2, "%s: rirb count reached\n", __FUNCTION__);

            return;

        }



        rp = (d->corb_rp + 1) & 0xff;

        addr = intel_hda_addr(d->corb_lbase, d->corb_ubase);

        verb = ldl_le_pci_dma(&d->pci, addr + 4*rp);

        d->corb_rp = rp;



        dprint(d, 2, "%s: [rp 0x%x] verb 0x%08x\n", __FUNCTION__, rp, verb);

        intel_hda_send_command(d, verb);

    }

}
