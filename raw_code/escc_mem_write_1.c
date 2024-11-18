static void escc_mem_write(void *opaque, hwaddr addr,

                           uint64_t val, unsigned size)

{

    ESCCState *serial = opaque;

    ChannelState *s;

    uint32_t saddr;

    int newreg, channel;



    val &= 0xff;

    saddr = (addr >> serial->it_shift) & 1;

    channel = (addr >> (serial->it_shift + 1)) & 1;

    s = &serial->chn[channel];

    switch (saddr) {

    case SERIAL_CTRL:

        trace_escc_mem_writeb_ctrl(CHN_C(s), s->reg, val & 0xff);

        newreg = 0;

        switch (s->reg) {

        case W_CMD:

            newreg = val & CMD_PTR_MASK;

            val &= CMD_CMD_MASK;

            switch (val) {

            case CMD_HI:

                newreg |= CMD_HI;

                break;

            case CMD_CLR_TXINT:

                clr_txint(s);

                break;

            case CMD_CLR_IUS:

                if (s->rxint_under_svc) {

                    s->rxint_under_svc = 0;

                    if (s->txint) {

                        set_txint(s);

                    }

                } else if (s->txint_under_svc) {

                    s->txint_under_svc = 0;

                }

                escc_update_irq(s);

                break;

            default:

                break;

            }

            break;

        case W_INTR ... W_RXCTRL:

        case W_SYNC1 ... W_TXBUF:

        case W_MISC1 ... W_CLOCK:

        case W_MISC2 ... W_EXTINT:

            s->wregs[s->reg] = val;

            break;

        case W_TXCTRL1:

        case W_TXCTRL2:

            s->wregs[s->reg] = val;

            escc_update_parameters(s);

            break;

        case W_BRGLO:

        case W_BRGHI:

            s->wregs[s->reg] = val;

            s->rregs[s->reg] = val;

            escc_update_parameters(s);

            break;

        case W_MINTR:

            switch (val & MINTR_RST_MASK) {

            case 0:

            default:

                break;

            case MINTR_RST_B:

                escc_reset_chn(&serial->chn[0]);

                return;

            case MINTR_RST_A:

                escc_reset_chn(&serial->chn[1]);

                return;

            case MINTR_RST_ALL:

                escc_reset(DEVICE(serial));

                return;

            }

            break;

        default:

            break;

        }

        if (s->reg == 0)

            s->reg = newreg;

        else

            s->reg = 0;

        break;

    case SERIAL_DATA:

        trace_escc_mem_writeb_data(CHN_C(s), val);

        s->tx = val;

        if (s->wregs[W_TXCTRL2] & TXCTRL2_TXEN) { // tx enabled

            if (s->chr)

                qemu_chr_fe_write(s->chr, &s->tx, 1);

            else if (s->type == kbd && !s->disabled) {

                handle_kbd_command(s, val);

            }

        }

        s->rregs[R_STATUS] |= STATUS_TXEMPTY; // Tx buffer empty

        s->rregs[R_SPEC] |= SPEC_ALLSENT; // All sent

        set_txint(s);

        break;

    default:

        break;

    }

}
