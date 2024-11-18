static void vfio_probe_rtl8168_bar2_window_quirk(VFIOPCIDevice *vdev, int nr)

{

    PCIDevice *pdev = &vdev->pdev;

    VFIOQuirk *quirk;



    if (pci_get_word(pdev->config + PCI_VENDOR_ID) != PCI_VENDOR_ID_REALTEK ||

        pci_get_word(pdev->config + PCI_DEVICE_ID) != 0x8168 || nr != 2) {

        return;

    }



    quirk = g_malloc0(sizeof(*quirk));

    quirk->vdev = vdev;

    quirk->data.bar = nr;



    memory_region_init_io(&quirk->mem, OBJECT(vdev), &vfio_rtl8168_window_quirk,

                          quirk, "vfio-rtl8168-window-quirk", 8);

    memory_region_add_subregion_overlap(&vdev->bars[nr].region.mem,

                                        0x70, &quirk->mem, 1);



    QLIST_INSERT_HEAD(&vdev->bars[nr].quirks, quirk, next);



    trace_vfio_probe_rtl8168_bar2_window_quirk(vdev->vbasedev.name);

}
