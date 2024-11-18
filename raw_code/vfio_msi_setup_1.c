static int vfio_msi_setup(VFIOPCIDevice *vdev, int pos)

{

    uint16_t ctrl;

    bool msi_64bit, msi_maskbit;

    int ret, entries;



    if (pread(vdev->vbasedev.fd, &ctrl, sizeof(ctrl),

              vdev->config_offset + pos + PCI_CAP_FLAGS) != sizeof(ctrl)) {

        return -errno;

    }

    ctrl = le16_to_cpu(ctrl);



    msi_64bit = !!(ctrl & PCI_MSI_FLAGS_64BIT);

    msi_maskbit = !!(ctrl & PCI_MSI_FLAGS_MASKBIT);

    entries = 1 << ((ctrl & PCI_MSI_FLAGS_QMASK) >> 1);



    trace_vfio_msi_setup(vdev->vbasedev.name, pos);



    ret = msi_init(&vdev->pdev, pos, entries, msi_64bit, msi_maskbit);

    if (ret < 0) {

        if (ret == -ENOTSUP) {

            return 0;

        }

        error_report("vfio: msi_init failed");

        return ret;

    }

    vdev->msi_cap_size = 0xa + (msi_maskbit ? 0xa : 0) + (msi_64bit ? 0x4 : 0);



    return 0;

}
