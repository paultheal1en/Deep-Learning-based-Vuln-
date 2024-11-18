uart_write(void *opaque, hwaddr addr,

           uint64_t val64, unsigned int size)

{

    XilinxUARTLite *s = opaque;

    uint32_t value = val64;

    unsigned char ch = value;



    addr >>= 2;

    switch (addr)

    {

        case R_STATUS:

            hw_error("write to UART STATUS?\n");

            break;



        case R_CTRL:

            if (value & CONTROL_RST_RX) {

                s->rx_fifo_pos = 0;

                s->rx_fifo_len = 0;

            }

            s->regs[addr] = value;

            break;



        case R_TX:

            if (s->chr)

                qemu_chr_fe_write(s->chr, &ch, 1);



            s->regs[addr] = value;



            /* hax.  */

            s->regs[R_STATUS] |= STATUS_IE;

            break;



        default:

            DUART(printf("%s addr=%x v=%x\n", __func__, addr, value));

            if (addr < ARRAY_SIZE(s->regs))

                s->regs[addr] = value;

            break;

    }

    uart_update_status(s);

    uart_update_irq(s);

}
