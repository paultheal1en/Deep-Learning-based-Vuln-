static void qvirtio_9p_pci_stop(QVirtIO9P *v9p)

{

    qvirtqueue_cleanup(v9p->dev->bus, v9p->vq, v9p->qs->alloc);

    qvirtio_pci_device_disable(container_of(v9p->dev, QVirtioPCIDevice, vdev));

    g_free(v9p->dev);

    qvirtio_9p_stop(v9p);

}
