static bool sdhci_can_issue_command(SDHCIState *s)

{

    if (!SDHC_CLOCK_IS_ON(s->clkcon) || !(s->pwrcon & SDHC_POWER_ON) ||

        (((s->prnsts & SDHC_DATA_INHIBIT) || s->stopped_state) &&

        ((s->cmdreg & SDHC_CMD_DATA_PRESENT) ||

        ((s->cmdreg & SDHC_CMD_RESPONSE) == SDHC_CMD_RSP_WITH_BUSY &&

        !(SDHC_COMMAND_TYPE(s->cmdreg) == SDHC_CMD_ABORT))))) {

        return false;

    }



    return true;

}
