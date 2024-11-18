static inline void IRQ_setbit(IRQQueue *q, int n_IRQ)

{

    set_bit(q->queue, n_IRQ);

}
