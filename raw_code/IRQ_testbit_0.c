static inline int IRQ_testbit(IRQQueue *q, int n_IRQ)

{

    return test_bit(q->queue, n_IRQ);

}
