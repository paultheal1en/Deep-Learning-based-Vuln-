static void vfio_pci_write_config(PCIDevice *pdev, uint32_t addr,

                                  uint32_t val, int len)

{

    VFIODevice *vdev = DO_UPCAST(VFIODevice, pdev, pdev);

    uint32_t val_le = cpu_to_le32(val);



    DPRINTF("%s(%04x:%02x:%02x.%x, @0x%x, 0x%x, len=0x%x)\n", __func__,

            vdev->host.domain, vdev->host.bus, vdev->host.slot,

            vdev->host.function, addr, val, len);



    /* Write everything to VFIO, let it filter out what we can't write */

    if (pwrite(vdev->fd, &val_le, len, vdev->config_offset + addr) != len) {

        error_report("%s(%04x:%02x:%02x.%x, 0x%x, 0x%x, 0x%x) failed: %m",

                     __func__, vdev->host.domain, vdev->host.bus,

                     vdev->host.slot, vdev->host.function, addr, val, len);

    }



    /* Write standard header bits to emulation */

    if (addr < PCI_CONFIG_HEADER_SIZE) {

        pci_default_write_config(pdev, addr, val, len);

        return;

    }



    /* MSI/MSI-X Enabling/Disabling */

    if (pdev->cap_present & QEMU_PCI_CAP_MSI &&

        ranges_overlap(addr, len, pdev->msi_cap, vdev->msi_cap_size)) {

        int is_enabled, was_enabled = msi_enabled(pdev);



        pci_default_write_config(pdev, addr, val, len);



        is_enabled = msi_enabled(pdev);



        if (!was_enabled && is_enabled) {

            vfio_enable_msi(vdev);

        } else if (was_enabled && !is_enabled) {

            vfio_disable_msi(vdev);

        }

    }



    if (pdev->cap_present & QEMU_PCI_CAP_MSIX &&

        ranges_overlap(addr, len, pdev->msix_cap, MSIX_CAP_LENGTH)) {

        int is_enabled, was_enabled = msix_enabled(pdev);



        pci_default_write_config(pdev, addr, val, len);



        is_enabled = msix_enabled(pdev);



        if (!was_enabled && is_enabled) {

            vfio_enable_msix(vdev);

        } else if (was_enabled && !is_enabled) {

            vfio_disable_msix(vdev);

        }

    }

}
