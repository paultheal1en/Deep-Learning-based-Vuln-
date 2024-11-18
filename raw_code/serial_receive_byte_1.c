static void serial_receive_byte(SerialState *s, int ch)

{

    s->rbr = ch;

    s->lsr |= UART_LSR_DR;

    serial_update_irq(s);

}
