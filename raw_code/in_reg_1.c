static inline unsigned in_reg(IVState *s, enum Reg reg)

{

    const char *name = reg2str(reg);

    QTestState *qtest = global_qtest;

    unsigned res;



    global_qtest = s->qtest;

    res = qpci_io_readl(s->dev, s->reg_base + reg);

    g_test_message("*%s -> %x\n", name, res);

    global_qtest = qtest;



    return res;

}
