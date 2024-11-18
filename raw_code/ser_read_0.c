ser_read(void *opaque, target_phys_addr_t addr, unsigned int size)

{

    struct etrax_serial *s = opaque;

    D(CPUCRISState *env = s->env);

    uint32_t r = 0;



    addr >>= 2;

    switch (addr)

    {

        case R_STAT_DIN:

            r = s->rx_fifo[(s->rx_fifo_pos - s->rx_fifo_len) & 15];

            if (s->rx_fifo_len) {

                r |= 1 << STAT_DAV;

            }

            r |= 1 << STAT_TR_RDY;

            r |= 1 << STAT_TR_IDLE;

            break;

        case RS_STAT_DIN:

            r = s->rx_fifo[(s->rx_fifo_pos - s->rx_fifo_len) & 15];

            if (s->rx_fifo_len) {

                r |= 1 << STAT_DAV;

                s->rx_fifo_len--;

            }

            r |= 1 << STAT_TR_RDY;

            r |= 1 << STAT_TR_IDLE;

            break;

        default:

            r = s->regs[addr];

            D(qemu_log("%s " TARGET_FMT_plx "=%x\n", __func__, addr, r));

            break;

    }

    return r;

}
