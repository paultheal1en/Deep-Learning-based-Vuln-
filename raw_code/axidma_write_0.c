static void axidma_write(void *opaque, hwaddr addr,

                         uint64_t value, unsigned size)

{

    XilinxAXIDMA *d = opaque;

    struct Stream *s;

    int sid;



    sid = streamid_from_addr(addr);

    s = &d->streams[sid];



    addr = addr % 0x30;

    addr >>= 2;

    switch (addr) {

        case R_DMACR:

            /* Tailptr mode is always on.  */

            value |= DMACR_TAILPTR_MODE;

            /* Remember our previous reset state.  */

            value |= (s->regs[addr] & DMACR_RESET);

            s->regs[addr] = value;



            if (value & DMACR_RESET) {

                stream_reset(s);

            }



            if ((value & 1) && !stream_resetting(s)) {

                /* Start processing.  */

                s->regs[R_DMASR] &= ~(DMASR_HALTED | DMASR_IDLE);

            }

            stream_reload_complete_cnt(s);

            break;



        case R_DMASR:

            /* Mask away write to clear irq lines.  */

            value &= ~(value & DMASR_IRQ_MASK);

            s->regs[addr] = value;

            break;



        case R_TAILDESC:

            s->regs[addr] = value;

            s->regs[R_DMASR] &= ~DMASR_IDLE; /* Not idle.  */

            if (!sid) {

                stream_process_mem2s(s, d->tx_dev);

            }

            break;

        default:

            D(qemu_log("%s: ch=%d addr=" TARGET_FMT_plx " v=%x\n",

                  __func__, sid, addr * 4, (unsigned)value));

            s->regs[addr] = value;

            break;

    }

    if (sid == 1 && d->notify) {

        d->notify(d->notify_opaque);

        d->notify = NULL;

    }

    stream_update_irq(s);

}
