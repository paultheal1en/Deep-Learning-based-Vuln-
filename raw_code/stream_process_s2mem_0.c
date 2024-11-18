static size_t stream_process_s2mem(struct Stream *s, unsigned char *buf,

                                   size_t len, uint32_t *app)

{

    uint32_t prev_d;

    unsigned int rxlen;

    size_t pos = 0;

    int sof = 1;



    if (!stream_running(s) || stream_idle(s)) {

        return 0;

    }



    while (len) {

        stream_desc_load(s, s->regs[R_CURDESC]);



        if (s->desc.status & SDESC_STATUS_COMPLETE) {

            s->regs[R_DMASR] |= DMASR_HALTED;

            break;

        }



        rxlen = s->desc.control & SDESC_CTRL_LEN_MASK;

        if (rxlen > len) {

            /* It fits.  */

            rxlen = len;

        }



        cpu_physical_memory_write(s->desc.buffer_address, buf + pos, rxlen);

        len -= rxlen;

        pos += rxlen;



        /* Update the descriptor.  */

        if (!len) {

            int i;



            stream_complete(s);

            for (i = 0; i < 5; i++) {

                s->desc.app[i] = app[i];

            }

            s->desc.status |= SDESC_STATUS_EOF;

        }



        s->desc.status |= sof << SDESC_STATUS_SOF_BIT;

        s->desc.status |= SDESC_STATUS_COMPLETE;

        stream_desc_store(s, s->regs[R_CURDESC]);

        sof = 0;



        /* Advance.  */

        prev_d = s->regs[R_CURDESC];

        s->regs[R_CURDESC] = s->desc.nxtdesc;

        if (prev_d == s->regs[R_TAILDESC]) {

            s->regs[R_DMASR] |= DMASR_IDLE;

            break;

        }

    }



    return pos;

}
