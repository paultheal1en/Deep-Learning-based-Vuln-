static void uart_write(void *opaque, hwaddr addr,

                       uint64_t value, unsigned size)

{

    LM32UartState *s = opaque;

    unsigned char ch = value;



    trace_lm32_uart_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_RXTX:

        if (s->chr) {



            qemu_chr_fe_write_all(s->chr, &ch, 1);

        }

        break;

    case R_IER:

    case R_LCR:

    case R_MCR:

    case R_DIV:

        s->regs[addr] = value;

        break;

    case R_IIR:

    case R_LSR:

    case R_MSR:

        error_report("lm32_uart: write access to read only register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    default:

        error_report("lm32_uart: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

    uart_update_irq(s);

}