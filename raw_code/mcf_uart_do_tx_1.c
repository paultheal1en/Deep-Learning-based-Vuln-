static void mcf_uart_do_tx(mcf_uart_state *s)

{

    if (s->tx_enabled && (s->sr & MCF_UART_TxEMP) == 0) {

        if (s->chr)

            qemu_chr_fe_write(s->chr, (unsigned char *)&s->tb, 1);

        s->sr |= MCF_UART_TxEMP;

    }

    if (s->tx_enabled) {

        s->sr |= MCF_UART_TxRDY;

    } else {

        s->sr &= ~MCF_UART_TxRDY;

    }

}
