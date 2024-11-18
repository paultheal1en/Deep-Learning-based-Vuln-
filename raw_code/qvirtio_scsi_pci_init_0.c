static QVirtIOSCSI *qvirtio_scsi_pci_init(int slot)

{

    QVirtIOSCSI *vs;

    QVirtioPCIDevice *dev;

    void *addr;

    int i;



    vs = g_new0(QVirtIOSCSI, 1);

    vs->alloc = pc_alloc_init();

    vs->bus = qpci_init_pc();



    dev = qvirtio_pci_device_find(vs->bus, QVIRTIO_SCSI_DEVICE_ID);

    vs->dev = (QVirtioDevice *)dev;

    g_assert(dev != NULL);

    g_assert_cmphex(vs->dev->device_type, ==, QVIRTIO_SCSI_DEVICE_ID);



    qvirtio_pci_device_enable(dev);

    qvirtio_reset(&qvirtio_pci, vs->dev);

    qvirtio_set_acknowledge(&qvirtio_pci, vs->dev);

    qvirtio_set_driver(&qvirtio_pci, vs->dev);



    addr = dev->addr + QVIRTIO_PCI_DEVICE_SPECIFIC_NO_MSIX;

    vs->num_queues = qvirtio_config_readl(&qvirtio_pci, vs->dev,

                                          (uint64_t)(uintptr_t)addr);



    g_assert_cmpint(vs->num_queues, <, MAX_NUM_QUEUES);



    for (i = 0; i < vs->num_queues + 2; i++) {

        vs->vq[i] = qvirtqueue_setup(&qvirtio_pci, vs->dev, vs->alloc, i);

    }



    return vs;

}
