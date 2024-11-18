static void test_tco1_status_bits(void)

{

    TestData d;

    uint16_t ticks = 8;

    uint16_t val;

    int ret;



    d.args = NULL;

    d.noreboot = true;

    test_init(&d);



    stop_tco(&d);

    clear_tco_status(&d);

    reset_on_second_timeout(false);

    set_tco_timeout(&d, ticks);

    load_tco(&d);

    start_tco(&d);

    clock_step(ticks * TCO_TICK_NSEC);



    qpci_io_writeb(d.dev, d.tco_io_base + TCO_DAT_IN, 0);

    qpci_io_writeb(d.dev, d.tco_io_base + TCO_DAT_OUT, 0);

    val = qpci_io_readw(d.dev, d.tco_io_base + TCO1_STS);

    ret = val & (TCO_TIMEOUT | SW_TCO_SMI | TCO_INT_STS) ? 1 : 0;

    g_assert(ret == 1);

    qpci_io_writew(d.dev, d.tco_io_base + TCO1_STS, val);

    g_assert_cmpint(qpci_io_readw(d.dev, d.tco_io_base + TCO1_STS), ==, 0);

    qtest_end();

}
