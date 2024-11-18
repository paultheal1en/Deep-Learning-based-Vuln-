static inline void set_txint(ChannelState *s)

{

    s->txint = 1;

    if (!s->rxint_under_svc) {

        s->txint_under_svc = 1;

        if (s->chn == chn_a) {

            s->rregs[R_INTR] |= INTR_TXINTA;

            if (s->wregs[W_MINTR] & MINTR_STATUSHI)

                s->otherchn->rregs[R_IVEC] = IVEC_HITXINTA;

            else

                s->otherchn->rregs[R_IVEC] = IVEC_LOTXINTA;

        } else {

            s->rregs[R_IVEC] = IVEC_TXINTB;

            s->otherchn->rregs[R_INTR] |= INTR_TXINTB;

        }

    escc_update_irq(s);

    }

}
