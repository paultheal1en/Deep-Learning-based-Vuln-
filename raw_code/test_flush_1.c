static void test_flush(void)

{

    QPCIDevice *dev;

    QPCIBar bmdma_bar, ide_bar;

    uint8_t data;



    ide_test_start(

        "-drive file=blkdebug::%s,if=ide,cache=writeback,format=raw",

        tmp_path);



    dev = get_pci_device(&bmdma_bar, &ide_bar);



    qtest_irq_intercept_in(global_qtest, "ioapic");



    /* Dirty media so that CMD_FLUSH_CACHE will actually go to disk */

    make_dirty(0);



    /* Delay the completion of the flush request until we explicitly do it */

    g_free(hmp("qemu-io ide0-hd0 \"break flush_to_os A\""));



    /* FLUSH CACHE command on device 0*/

    qpci_io_writeb(dev, ide_bar, reg_device, 0);

    qpci_io_writeb(dev, ide_bar, reg_command, CMD_FLUSH_CACHE);



    /* Check status while request is in flight*/

    data = qpci_io_readb(dev, ide_bar, reg_status);

    assert_bit_set(data, BSY | DRDY);

    assert_bit_clear(data, DF | ERR | DRQ);



    /* Complete the command */

    g_free(hmp("qemu-io ide0-hd0 \"resume A\""));



    /* Check registers */

    data = qpci_io_readb(dev, ide_bar, reg_device);

    g_assert_cmpint(data & DEV, ==, 0);



    do {

        data = qpci_io_readb(dev, ide_bar, reg_status);

    } while (data & BSY);



    assert_bit_set(data, DRDY);

    assert_bit_clear(data, BSY | DF | ERR | DRQ);



    ide_test_quit();


}