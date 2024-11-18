static void vfio_msi_interrupt(void *opaque)

{

    VFIOMSIVector *vector = opaque;

    VFIODevice *vdev = vector->vdev;

    int nr = vector - vdev->msi_vectors;



    if (!event_notifier_test_and_clear(&vector->interrupt)) {

        return;

    }



    DPRINTF("%s(%04x:%02x:%02x.%x) vector %d\n", __func__,

            vdev->host.domain, vdev->host.bus, vdev->host.slot,

            vdev->host.function, nr);



    if (vdev->interrupt == VFIO_INT_MSIX) {

        msix_notify(&vdev->pdev, nr);

    } else if (vdev->interrupt == VFIO_INT_MSI) {

        msi_notify(&vdev->pdev, nr);

    } else {

        error_report("vfio: MSI interrupt receieved, but not enabled?");

    }

}
