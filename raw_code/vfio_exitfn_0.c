static void vfio_exitfn(PCIDevice *pdev)

{

    VFIOPCIDevice *vdev = DO_UPCAST(VFIOPCIDevice, pdev, pdev);



    vfio_unregister_err_notifier(vdev);

    pci_device_set_intx_routing_notifier(&vdev->pdev, NULL);

    vfio_disable_interrupts(vdev);

    if (vdev->intx.mmap_timer) {

        timer_free(vdev->intx.mmap_timer);

    }

    vfio_teardown_msi(vdev);

    vfio_unmap_bars(vdev);

}
