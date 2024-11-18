static void qvirtio_9p_pci_free(QVirtIO9P *v9p)

{

    qvirtqueue_cleanup(v9p->dev->bus, v9p->vq, v9p->alloc);

    pc_alloc_uninit(v9p->alloc);

    qvirtio_pci_device_disable(container_of(v9p->dev, QVirtioPCIDevice, vdev));

    g_free(v9p->dev);

    qpci_free_pc(v9p->bus);

    g_free(v9p);

}
