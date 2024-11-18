static void uart_write(void *opaque, hwaddr offset,

                          uint64_t value, unsigned size)

{

    CadenceUARTState *s = opaque;



    DB_PRINT(" offset:%x data:%08x\n", (unsigned)offset, (unsigned)value);

    offset >>= 2;




    switch (offset) {

    case R_IER: /* ier (wts imr) */

        s->r[R_IMR] |= value;

        break;

    case R_IDR: /* idr (wtc imr) */

        s->r[R_IMR] &= ~value;

        break;

    case R_IMR: /* imr (read only) */

        break;

    case R_CISR: /* cisr (wtc) */

        s->r[R_CISR] &= ~value;

        break;

    case R_TX_RX: /* UARTDR */

        switch (s->r[R_MR] & UART_MR_CHMODE) {

        case NORMAL_MODE:

            uart_write_tx_fifo(s, (uint8_t *) &value, 1);

            break;

        case LOCAL_LOOPBACK:

            uart_write_rx_fifo(opaque, (uint8_t *) &value, 1);

            break;


        break;

    default:

        s->r[offset] = value;




    switch (offset) {

    case R_CR:

        uart_ctrl_update(s);

        break;

    case R_MR:

        uart_parameters_setup(s);

        break;


    uart_update_status(s);
