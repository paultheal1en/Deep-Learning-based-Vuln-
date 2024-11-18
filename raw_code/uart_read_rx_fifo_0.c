static void uart_read_rx_fifo(UartState *s, uint32_t *c)

{

    if ((s->r[R_CR] & UART_CR_RX_DIS) || !(s->r[R_CR] & UART_CR_RX_EN)) {

        return;

    }



    if (s->rx_count) {

        uint32_t rx_rpos =

                (RX_FIFO_SIZE + s->rx_wpos - s->rx_count) % RX_FIFO_SIZE;

        *c = s->rx_fifo[rx_rpos];

        s->rx_count--;



        qemu_chr_accept_input(s->chr);

    } else {

        *c = 0;

    }



    uart_update_status(s);

}
