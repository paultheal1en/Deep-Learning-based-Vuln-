static inline int IRQ_testbit(IRQ_queue_t *q, int n_IRQ)

{

    return test_bit(q->queue, n_IRQ);

}
