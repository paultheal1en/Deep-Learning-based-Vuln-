static void fifo_trigger_update(void *opaque)

{

    CadenceUARTState *s = opaque;



    s->r[R_CISR] |= UART_INTR_TIMEOUT;



    uart_update_status(s);

}
