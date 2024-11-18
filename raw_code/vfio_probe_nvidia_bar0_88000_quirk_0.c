static void vfio_probe_nvidia_bar0_88000_quirk(VFIODevice *vdev, int nr)

{

    PCIDevice *pdev = &vdev->pdev;

    VFIOQuirk *quirk;



    if (!vdev->has_vga || nr != 0 ||

        pci_get_word(pdev->config + PCI_VENDOR_ID) != PCI_VENDOR_ID_NVIDIA) {

        return;

    }



    quirk = g_malloc0(sizeof(*quirk));

    quirk->vdev = vdev;

    quirk->data.flags = quirk->data.read_flags = quirk->data.write_flags = 1;

    quirk->data.address_match = 0x88000;

    quirk->data.address_mask = PCIE_CONFIG_SPACE_SIZE - 1;

    quirk->data.bar = nr;



    memory_region_init_io(&quirk->mem, OBJECT(vdev), &vfio_generic_quirk,

                          quirk, "vfio-nvidia-bar0-88000-quirk",

                          TARGET_PAGE_ALIGN(quirk->data.address_mask + 1));

    memory_region_add_subregion_overlap(&vdev->bars[nr].mem,

                          quirk->data.address_match & TARGET_PAGE_MASK,

                          &quirk->mem, 1);



    QLIST_INSERT_HEAD(&vdev->bars[nr].quirks, quirk, next);



    DPRINTF("Enabled NVIDIA BAR0 0x88000 quirk for device %04x:%02x:%02x.%x\n",

            vdev->host.domain, vdev->host.bus, vdev->host.slot,

            vdev->host.function);

}
