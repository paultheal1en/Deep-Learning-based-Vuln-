static void test_tco_defaults(void)

{

    TestData d;



    d.args = NULL;

    d.noreboot = true;

    test_init(&d);

    g_assert_cmpint(qpci_io_readw(d.dev, d.tco_io_bar, TCO_RLD), ==,

                    TCO_RLD_DEFAULT);

    /* TCO_DAT_IN & TCO_DAT_OUT */

    g_assert_cmpint(qpci_io_readw(d.dev, d.tco_io_bar, TCO_DAT_IN), ==,

                    (TCO_DAT_OUT_DEFAULT << 8) | TCO_DAT_IN_DEFAULT);

    /* TCO1_STS & TCO2_STS */

    g_assert_cmpint(qpci_io_readl(d.dev, d.tco_io_bar, TCO1_STS), ==,

                    (TCO2_STS_DEFAULT << 16) | TCO1_STS_DEFAULT);

    /* TCO1_CNT & TCO2_CNT */

    g_assert_cmpint(qpci_io_readl(d.dev, d.tco_io_bar, TCO1_CNT), ==,

                    (TCO2_CNT_DEFAULT << 16) | TCO1_CNT_DEFAULT);

    /* TCO_MESSAGE1 & TCO_MESSAGE2 */

    g_assert_cmpint(qpci_io_readw(d.dev, d.tco_io_bar, TCO_MESSAGE1), ==,

                    (TCO_MESSAGE2_DEFAULT << 8) | TCO_MESSAGE1_DEFAULT);

    g_assert_cmpint(qpci_io_readb(d.dev, d.tco_io_bar, TCO_WDCNT), ==,

                    TCO_WDCNT_DEFAULT);

    g_assert_cmpint(qpci_io_readb(d.dev, d.tco_io_bar, SW_IRQ_GEN), ==,

                    SW_IRQ_GEN_DEFAULT);

    g_assert_cmpint(qpci_io_readw(d.dev, d.tco_io_bar, TCO_TMR), ==,

                    TCO_TMR_DEFAULT);

    qtest_end();

}
