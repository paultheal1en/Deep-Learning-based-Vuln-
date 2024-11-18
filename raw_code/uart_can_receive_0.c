static int uart_can_receive(void *opaque)

{

    UartState *s = (UartState *)opaque;



    return RX_FIFO_SIZE - s->rx_count;

}
