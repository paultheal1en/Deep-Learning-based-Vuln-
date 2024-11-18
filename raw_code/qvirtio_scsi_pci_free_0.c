static void qvirtio_scsi_pci_free(QVirtIOSCSI *vs)

{

    int i;



    for (i = 0; i < vs->num_queues + 2; i++) {

        qvirtqueue_cleanup(vs->dev->bus, vs->vq[i], vs->alloc);

    }

    pc_alloc_uninit(vs->alloc);

    qvirtio_pci_device_disable(container_of(vs->dev, QVirtioPCIDevice, vdev));

    g_free(vs->dev);

    qpci_free_pc(vs->bus);

    g_free(vs);

}
