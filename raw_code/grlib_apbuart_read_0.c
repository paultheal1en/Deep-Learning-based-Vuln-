static uint64_t grlib_apbuart_read(void *opaque, target_phys_addr_t addr,

                                   unsigned size)

{

    UART     *uart = opaque;



    addr &= 0xff;



    /* Unit registers */

    switch (addr) {

    case DATA_OFFSET:

    case DATA_OFFSET + 3:       /* when only one byte read */

        return uart_pop(uart);



    case STATUS_OFFSET:

        /* Read Only */

        return uart->status;



    case CONTROL_OFFSET:

        return uart->control;



    case SCALER_OFFSET:

        /* Not supported */

        return 0;



    default:

        trace_grlib_apbuart_readl_unknown(addr);

        return 0;

    }

}
