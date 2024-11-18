static inline void read_mem(IVState *s, uint64_t off, void *buf, size_t len)

{

    QTestState *qtest = global_qtest;



    global_qtest = s->qtest;

    qpci_memread(s->dev, s->mem_base + off, buf, len);

    global_qtest = qtest;

}
