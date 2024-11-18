static void openpic_tmr_write(void *opaque, hwaddr addr, uint64_t val,

                                unsigned len)

{

    OpenPICState *opp = opaque;

    int idx;



    DPRINTF("%s: addr %08x <= %08x\n", __func__, addr, val);

    if (addr & 0xF)

        return;

    idx = (addr >> 6) & 0x3;

    addr = addr & 0x30;



    if (addr == 0x0) {

        /* TIFR (TFRR) */

        opp->tifr = val;

        return;

    }

    switch (addr & 0x30) {

    case 0x00: /* TICC (GTCCR) */

        break;

    case 0x10: /* TIBC (GTBCR) */

        if ((opp->timers[idx].ticc & TICC_TOG) != 0 &&

            (val & TIBC_CI) == 0 &&

            (opp->timers[idx].tibc & TIBC_CI) != 0) {

            opp->timers[idx].ticc &= ~TICC_TOG;

        }

        opp->timers[idx].tibc = val;

        break;

    case 0x20: /* TIVP (GTIVPR) */

        write_IRQreg_ipvp(opp, opp->irq_tim0 + idx, val);

        break;

    case 0x30: /* TIDE (GTIDR) */

        write_IRQreg_ide(opp, opp->irq_tim0 + idx, val);

        break;

    }

}
