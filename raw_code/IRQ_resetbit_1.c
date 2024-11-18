static inline void IRQ_resetbit(IRQ_queue_t *q, int n_IRQ)

{

    q->pending--;

    reset_bit(q->queue, n_IRQ);

}
