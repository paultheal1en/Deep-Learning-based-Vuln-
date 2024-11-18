static QVirtIO9P *qvirtio_9p_pci_init(void)

{

    QVirtIO9P *v9p;

    QVirtioPCIDevice *dev;



    v9p = g_new0(QVirtIO9P, 1);

    v9p->alloc = pc_alloc_init();

    v9p->bus = qpci_init_pc(NULL);



    dev = qvirtio_pci_device_find(v9p->bus, VIRTIO_ID_9P);

    g_assert_nonnull(dev);

    g_assert_cmphex(dev->vdev.device_type, ==, VIRTIO_ID_9P);

    v9p->dev = (QVirtioDevice *) dev;



    qvirtio_pci_device_enable(dev);

    qvirtio_reset(v9p->dev);

    qvirtio_set_acknowledge(v9p->dev);

    qvirtio_set_driver(v9p->dev);



    v9p->vq = qvirtqueue_setup(v9p->dev, v9p->alloc, 0);

    return v9p;

}
