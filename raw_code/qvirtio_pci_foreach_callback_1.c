static void qvirtio_pci_foreach_callback(

                        QPCIDevice *dev, int devfn, void *data)

{

    QVirtioPCIForeachData *d = data;

    QVirtioPCIDevice *vpcidev = qpcidevice_to_qvirtiodevice(dev);



    if (vpcidev->vdev.device_type == d->device_type &&

        (!d->has_slot || vpcidev->pdev->devfn == d->slot << 3)) {

        d->func(&vpcidev->vdev, d->user_data);

    } else {

        g_free(vpcidev);

    }

}
