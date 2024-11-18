ser_write(void *opaque, hwaddr addr,

          uint64_t val64, unsigned int size)

{

    ETRAXSerial *s = opaque;

    uint32_t value = val64;

    unsigned char ch = val64;



    D(qemu_log("%s " TARGET_FMT_plx "=%x\n",  __func__, addr, value));

    addr >>= 2;

    switch (addr)

    {

        case RW_DOUT:

            qemu_chr_fe_write(s->chr, &ch, 1);

            s->regs[R_INTR] |= 3;

            s->pending_tx = 1;

            s->regs[addr] = value;

            break;

        case RW_ACK_INTR:

            if (s->pending_tx) {

                value &= ~1;

                s->pending_tx = 0;

                D(qemu_log("fixedup value=%x r_intr=%x\n",

                           value, s->regs[R_INTR]));

            }

            s->regs[addr] = value;

            s->regs[R_INTR] &= ~value;

            D(printf("r_intr=%x\n", s->regs[R_INTR]));

            break;

        default:

            s->regs[addr] = value;

            break;

    }

    ser_update_irq(s);

}
