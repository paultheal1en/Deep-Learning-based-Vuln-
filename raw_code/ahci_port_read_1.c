static uint32_t  ahci_port_read(AHCIState *s, int port, int offset)

{

    uint32_t val;

    AHCIPortRegs *pr;

    pr = &s->dev[port].port_regs;



    switch (offset) {

    case PORT_LST_ADDR:

        val = pr->lst_addr;

        break;

    case PORT_LST_ADDR_HI:

        val = pr->lst_addr_hi;

        break;

    case PORT_FIS_ADDR:

        val = pr->fis_addr;

        break;

    case PORT_FIS_ADDR_HI:

        val = pr->fis_addr_hi;

        break;

    case PORT_IRQ_STAT:

        val = pr->irq_stat;

        break;

    case PORT_IRQ_MASK:

        val = pr->irq_mask;

        break;

    case PORT_CMD:

        val = pr->cmd;

        break;

    case PORT_TFDATA:

        val = ((uint16_t)s->dev[port].port.ifs[0].error << 8) |

              s->dev[port].port.ifs[0].status;

        break;

    case PORT_SIG:

        val = pr->sig;

        break;

    case PORT_SCR_STAT:

        if (s->dev[port].port.ifs[0].bs) {

            val = SATA_SCR_SSTATUS_DET_DEV_PRESENT_PHY_UP |

                  SATA_SCR_SSTATUS_SPD_GEN1 | SATA_SCR_SSTATUS_IPM_ACTIVE;

        } else {

            val = SATA_SCR_SSTATUS_DET_NODEV;

        }

        break;

    case PORT_SCR_CTL:

        val = pr->scr_ctl;

        break;

    case PORT_SCR_ERR:

        val = pr->scr_err;

        break;

    case PORT_SCR_ACT:

        pr->scr_act &= ~s->dev[port].finished;

        s->dev[port].finished = 0;

        val = pr->scr_act;

        break;

    case PORT_CMD_ISSUE:

        val = pr->cmd_issue;

        break;

    case PORT_RESERVED:

    default:

        val = 0;

    }

    DPRINTF(port, "offset: 0x%x val: 0x%x\n", offset, val);

    return val;



}
