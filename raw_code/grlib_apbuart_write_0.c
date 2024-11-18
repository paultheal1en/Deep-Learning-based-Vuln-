static void grlib_apbuart_write(void *opaque, target_phys_addr_t addr,

                                uint64_t value, unsigned size)

{

    UART          *uart = opaque;

    unsigned char  c    = 0;



    addr &= 0xff;



    /* Unit registers */

    switch (addr) {

    case DATA_OFFSET:

    case DATA_OFFSET + 3:       /* When only one byte write */

        c = value & 0xFF;

        qemu_chr_fe_write(uart->chr, &c, 1);

        return;



    case STATUS_OFFSET:

        /* Read Only */

        return;



    case CONTROL_OFFSET:

        uart->control = value;

        return;



    case SCALER_OFFSET:

        /* Not supported */

        return;



    default:

        break;

    }



    trace_grlib_apbuart_writel_unknown(addr, value);

}
