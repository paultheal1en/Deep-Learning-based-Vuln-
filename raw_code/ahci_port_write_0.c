static void  ahci_port_write(AHCIState *s, int port, int offset, uint32_t val)

{

    AHCIPortRegs *pr = &s->dev[port].port_regs;



    DPRINTF(port, "offset: 0x%x val: 0x%x\n", offset, val);

    switch (offset) {

        case PORT_LST_ADDR:

            pr->lst_addr = val;

            break;

        case PORT_LST_ADDR_HI:

            pr->lst_addr_hi = val;

            break;

        case PORT_FIS_ADDR:

            pr->fis_addr = val;

            break;

        case PORT_FIS_ADDR_HI:

            pr->fis_addr_hi = val;

            break;

        case PORT_IRQ_STAT:

            pr->irq_stat &= ~val;

            ahci_check_irq(s);

            break;

        case PORT_IRQ_MASK:

            pr->irq_mask = val & 0xfdc000ff;

            ahci_check_irq(s);

            break;

        case PORT_CMD:

            pr->cmd = val & ~(PORT_CMD_LIST_ON | PORT_CMD_FIS_ON);



            if (pr->cmd & PORT_CMD_START) {

                if (ahci_map_clb_address(&s->dev[port])) {

                    pr->cmd |= PORT_CMD_LIST_ON;

                } else {

                    error_report("AHCI: Failed to start DMA engine: "

                                 "bad command list buffer address");

                }

            }



            if (pr->cmd & PORT_CMD_FIS_RX) {

                if (ahci_map_fis_address(&s->dev[port])) {

                    pr->cmd |= PORT_CMD_FIS_ON;

                } else {

                    error_report("AHCI: Failed to start FIS receive engine: "

                                 "bad FIS receive buffer address");

                }

            }



            /* XXX usually the FIS would be pending on the bus here and

                   issuing deferred until the OS enables FIS receival.

                   Instead, we only submit it once - which works in most

                   cases, but is a hack. */

            if ((pr->cmd & PORT_CMD_FIS_ON) &&

                !s->dev[port].init_d2h_sent) {

                ahci_init_d2h(&s->dev[port]);

                s->dev[port].init_d2h_sent = true;

            }



            check_cmd(s, port);

            break;

        case PORT_TFDATA:

            /* Read Only. */

            break;

        case PORT_SIG:

            /* Read Only */

            break;

        case PORT_SCR_STAT:

            /* Read Only */

            break;

        case PORT_SCR_CTL:

            if (((pr->scr_ctl & AHCI_SCR_SCTL_DET) == 1) &&

                ((val & AHCI_SCR_SCTL_DET) == 0)) {

                ahci_reset_port(s, port);

            }

            pr->scr_ctl = val;

            break;

        case PORT_SCR_ERR:

            pr->scr_err &= ~val;

            break;

        case PORT_SCR_ACT:

            /* RW1 */

            pr->scr_act |= val;

            break;

        case PORT_CMD_ISSUE:

            pr->cmd_issue |= val;

            check_cmd(s, port);

            break;

        default:

            break;

    }

}
