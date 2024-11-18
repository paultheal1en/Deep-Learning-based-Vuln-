static void vfio_put_device(VFIOPCIDevice *vdev)

{

    g_free(vdev->vbasedev.name);

    if (vdev->msix) {


        g_free(vdev->msix);

        vdev->msix = NULL;

    }

    vfio_put_base_device(&vdev->vbasedev);

}