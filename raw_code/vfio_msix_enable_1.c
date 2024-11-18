static void vfio_msix_enable(VFIOPCIDevice *vdev)

{

    vfio_disable_interrupts(vdev);



    vdev->msi_vectors = g_malloc0(vdev->msix->entries * sizeof(VFIOMSIVector));



    vdev->interrupt = VFIO_INT_MSIX;



    /*

     * Some communication channels between VF & PF or PF & fw rely on the

     * physical state of the device and expect that enabling MSI-X from the

     * guest enables the same on the host.  When our guest is Linux, the

     * guest driver call to pci_enable_msix() sets the enabling bit in the

     * MSI-X capability, but leaves the vector table masked.  We therefore

     * can't rely on a vector_use callback (from request_irq() in the guest)

     * to switch the physical device into MSI-X mode because that may come a

     * long time after pci_enable_msix().  This code enables vector 0 with

     * triggering to userspace, then immediately release the vector, leaving

     * the physical device with no vectors enabled, but MSI-X enabled, just

     * like the guest view.

     */

    vfio_msix_vector_do_use(&vdev->pdev, 0, NULL, NULL);

    vfio_msix_vector_release(&vdev->pdev, 0);



    if (msix_set_vector_notifiers(&vdev->pdev, vfio_msix_vector_use,

                                  vfio_msix_vector_release, NULL)) {

        error_report("vfio: msix_set_vector_notifiers failed");

    }



    trace_vfio_msix_enable(vdev->vbasedev.name);

}
