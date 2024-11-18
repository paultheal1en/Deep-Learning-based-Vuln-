static void pci_config(void)

{

    QVirtioPCIDevice *dev;

    QOSState *qs;

    int n_size = TEST_IMAGE_SIZE / 2;

    uint64_t capacity;



    qs = pci_test_start();



    dev = virtio_blk_pci_init(qs->pcibus, PCI_SLOT);



    capacity = qvirtio_config_readq(&dev->vdev, 0);

    g_assert_cmpint(capacity, ==, TEST_IMAGE_SIZE / 512);



    qvirtio_set_driver_ok(&dev->vdev);



    qmp_discard_response("{ 'execute': 'block_resize', "

                         " 'arguments': { 'device': 'drive0', "

                         " 'size': %d } }", n_size);

    qvirtio_wait_config_isr(&dev->vdev, QVIRTIO_BLK_TIMEOUT_US);



    capacity = qvirtio_config_readq(&dev->vdev, 0);

    g_assert_cmpint(capacity, ==, n_size / 512);



    qvirtio_pci_device_disable(dev);

    g_free(dev);



    qtest_shutdown(qs);

}
