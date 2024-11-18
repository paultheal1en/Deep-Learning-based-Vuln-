static void test_init(TestData *d)

{

    QPCIBus *bus;

    QTestState *qs;

    char *s;



    s = g_strdup_printf("-machine q35 %s", !d->args ? "" : d->args);

    qs = qtest_start(s);

    qtest_irq_intercept_in(qs, "ioapic");

    g_free(s);



    bus = qpci_init_pc();

    d->dev = qpci_device_find(bus, QPCI_DEVFN(0x1f, 0x00));

    g_assert(d->dev != NULL);



    /* map PCI-to-LPC bridge interface BAR */

    d->lpc_base = qpci_iomap(d->dev, 0, NULL);



    qpci_device_enable(d->dev);



    g_assert(d->lpc_base != NULL);



    /* set ACPI PM I/O space base address */

    qpci_config_writel(d->dev, (uintptr_t)d->lpc_base + ICH9_LPC_PMBASE,

                       PM_IO_BASE_ADDR | 0x1);

    /* enable ACPI I/O */

    qpci_config_writeb(d->dev, (uintptr_t)d->lpc_base + ICH9_LPC_ACPI_CTRL,

                       0x80);

    /* set Root Complex BAR */

    qpci_config_writel(d->dev, (uintptr_t)d->lpc_base + ICH9_LPC_RCBA,

                       RCBA_BASE_ADDR | 0x1);



    d->tco_io_base = (void *)((uintptr_t)PM_IO_BASE_ADDR + 0x60);

}
