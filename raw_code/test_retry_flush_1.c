static void test_retry_flush(const char *machine)

{

    QPCIDevice *dev;

    void *bmdma_base, *ide_base;

    uint8_t data;

    const char *s;



    prepare_blkdebug_script(debug_path, "flush_to_disk");



    ide_test_start(

        "-vnc none "

        "-drive file=blkdebug:%s:%s,if=ide,cache=writeback,format=raw,"

        "rerror=stop,werror=stop",

        debug_path, tmp_path);



    dev = get_pci_device(&bmdma_base, &ide_base);



    qtest_irq_intercept_in(global_qtest, "ioapic");



    /* Dirty media so that CMD_FLUSH_CACHE will actually go to disk */

    make_dirty(0);



    /* FLUSH CACHE command on device 0*/

    qpci_io_writeb(dev, ide_base + reg_device, 0);

    qpci_io_writeb(dev, ide_base + reg_command, CMD_FLUSH_CACHE);



    /* Check status while request is in flight*/

    data = qpci_io_readb(dev, ide_base + reg_status);

    assert_bit_set(data, BSY | DRDY);

    assert_bit_clear(data, DF | ERR | DRQ);



    qmp_eventwait("STOP");



    /* Complete the command */

    s = "{'execute':'cont' }";

    qmp_discard_response(s);



    /* Check registers */

    data = qpci_io_readb(dev, ide_base + reg_device);

    g_assert_cmpint(data & DEV, ==, 0);



    do {

        data = qpci_io_readb(dev, ide_base + reg_status);

    } while (data & BSY);



    assert_bit_set(data, DRDY);

    assert_bit_clear(data, BSY | DF | ERR | DRQ);



    ide_test_quit();

}
