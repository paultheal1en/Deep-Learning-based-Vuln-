static void uart_parameters_setup(UartState *s)

{

    QEMUSerialSetParams ssp;

    unsigned int baud_rate, packet_size;



    baud_rate = (s->r[R_MR] & UART_MR_CLKS) ?

            UART_INPUT_CLK / 8 : UART_INPUT_CLK;



    ssp.speed = baud_rate / (s->r[R_BRGR] * (s->r[R_BDIV] + 1));

    packet_size = 1;



    switch (s->r[R_MR] & UART_MR_PAR) {

    case UART_PARITY_EVEN:

        ssp.parity = 'E';

        packet_size++;

        break;

    case UART_PARITY_ODD:

        ssp.parity = 'O';

        packet_size++;

        break;

    default:

        ssp.parity = 'N';

        break;

    }



    switch (s->r[R_MR] & UART_MR_CHRL) {

    case UART_DATA_BITS_6:

        ssp.data_bits = 6;

        break;

    case UART_DATA_BITS_7:

        ssp.data_bits = 7;

        break;

    default:

        ssp.data_bits = 8;

        break;

    }



    switch (s->r[R_MR] & UART_MR_NBSTOP) {

    case UART_STOP_BITS_1:

        ssp.stop_bits = 1;

        break;

    default:

        ssp.stop_bits = 2;

        break;

    }



    packet_size += ssp.data_bits + ssp.stop_bits;

    s->char_tx_time = (get_ticks_per_sec() / ssp.speed) * packet_size;

    qemu_chr_fe_ioctl(s->chr, CHR_IOCTL_SERIAL_SET_PARAMS, &ssp);

}
