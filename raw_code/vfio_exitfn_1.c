static void vfio_exitfn(PCIDevice *pdev)

{

    VFIOPCIDevice *vdev = DO_UPCAST(VFIOPCIDevice, pdev, pdev);

    VFIOGroup *group = vdev->vbasedev.group;



    vfio_unregister_err_notifier(vdev);

    pci_device_set_intx_routing_notifier(&vdev->pdev, NULL);

    vfio_disable_interrupts(vdev);

    if (vdev->intx.mmap_timer) {

        timer_free(vdev->intx.mmap_timer);

    }

    vfio_teardown_msi(vdev);

    vfio_unmap_bars(vdev);

    g_free(vdev->emulated_config_bits);

    g_free(vdev->rom);

    vfio_put_device(vdev);

    vfio_put_group(group);

}
