static void setup_vm_cmd(IVState *s, const char *cmd, bool msix)

{

    uint64_t barsize;



    s->qtest = qtest_start(cmd);

    s->pcibus = qpci_init_pc(NULL);

    s->dev = get_device(s->pcibus);



    s->reg_base = qpci_iomap(s->dev, 0, &barsize);

    g_assert_nonnull(s->reg_base);

    g_assert_cmpuint(barsize, ==, 256);



    if (msix) {

        qpci_msix_enable(s->dev);

    }



    s->mem_base = qpci_iomap(s->dev, 2, &barsize);

    g_assert_nonnull(s->mem_base);

    g_assert_cmpuint(barsize, ==, TMPSHMSIZE);



    qpci_device_enable(s->dev);

}
