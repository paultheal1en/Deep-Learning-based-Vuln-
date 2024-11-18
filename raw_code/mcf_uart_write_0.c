void mcf_uart_write(void *opaque, target_phys_addr_t addr,

                    uint64_t val, unsigned size)

{

    mcf_uart_state *s = (mcf_uart_state *)opaque;

    switch (addr & 0x3f) {

    case 0x00:

        s->mr[s->current_mr] = val;

        s->current_mr = 1;

        break;

    case 0x04:

        /* CSR is ignored.  */

        break;

    case 0x08: /* Command Register.  */

        mcf_do_command(s, val);

        break;

    case 0x0c: /* Transmit Buffer.  */

        s->sr &= ~MCF_UART_TxEMP;

        s->tb = val;

        mcf_uart_do_tx(s);

        break;

    case 0x10:

        /* ACR is ignored.  */

        break;

    case 0x14:

        s->imr = val;

        break;

    default:

        break;

    }

    mcf_uart_update(s);

}
