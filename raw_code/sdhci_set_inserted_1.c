static void sdhci_set_inserted(DeviceState *dev, bool level)

{

    SDHCIState *s = (SDHCIState *)dev;

    DPRINT_L1("Card state changed: %s!\n", level ? "insert" : "eject");



    if ((s->norintsts & SDHC_NIS_REMOVE) && level) {

        /* Give target some time to notice card ejection */

        timer_mod(s->insert_timer,

                       qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + SDHC_INSERTION_DELAY);

    } else {

        if (level) {

            s->prnsts = 0x1ff0000;

            if (s->norintstsen & SDHC_NISEN_INSERT) {

                s->norintsts |= SDHC_NIS_INSERT;

            }

        } else {

            s->prnsts = 0x1fa0000;

            s->pwrcon &= ~SDHC_POWER_ON;

            s->clkcon &= ~SDHC_CLOCK_SDCLK_EN;

            if (s->norintstsen & SDHC_NISEN_REMOVE) {

                s->norintsts |= SDHC_NIS_REMOVE;

            }

        }

        sdhci_update_irq(s);

    }

}
