static void vfio_realize(PCIDevice *pdev, Error **errp)

{

    VFIOPCIDevice *vdev = DO_UPCAST(VFIOPCIDevice, pdev, pdev);

    VFIODevice *vbasedev_iter;

    VFIOGroup *group;

    char *tmp, group_path[PATH_MAX], *group_name;

    Error *err = NULL;

    ssize_t len;

    struct stat st;

    int groupid;

    int i, ret;



    if (!vdev->vbasedev.sysfsdev) {

        if (!(~vdev->host.domain || ~vdev->host.bus ||

              ~vdev->host.slot || ~vdev->host.function)) {

            error_setg(errp, "No provided host device");

            error_append_hint(errp, "Use -vfio-pci,host=DDDD:BB:DD.F "

                              "or -vfio-pci,sysfsdev=PATH_TO_DEVICE\n");

            return;

        }

        vdev->vbasedev.sysfsdev =

            g_strdup_printf("/sys/bus/pci/devices/%04x:%02x:%02x.%01x",

                            vdev->host.domain, vdev->host.bus,

                            vdev->host.slot, vdev->host.function);

    }



    if (stat(vdev->vbasedev.sysfsdev, &st) < 0) {

        error_setg_errno(errp, errno, "no such host device");

        error_prepend(errp, ERR_PREFIX, vdev->vbasedev.sysfsdev);

        return;

    }



    vdev->vbasedev.name = g_strdup(basename(vdev->vbasedev.sysfsdev));

    vdev->vbasedev.ops = &vfio_pci_ops;

    vdev->vbasedev.type = VFIO_DEVICE_TYPE_PCI;



    tmp = g_strdup_printf("%s/iommu_group", vdev->vbasedev.sysfsdev);

    len = readlink(tmp, group_path, sizeof(group_path));

    g_free(tmp);



    if (len <= 0 || len >= sizeof(group_path)) {

        error_setg_errno(errp, len < 0 ? errno : ENAMETOOLONG,

                         "no iommu_group found");

        goto error;

    }



    group_path[len] = 0;



    group_name = basename(group_path);

    if (sscanf(group_name, "%d", &groupid) != 1) {

        error_setg_errno(errp, errno, "failed to read %s", group_path);

        goto error;

    }



    trace_vfio_realize(vdev->vbasedev.name, groupid);



    group = vfio_get_group(groupid, pci_device_iommu_address_space(pdev), errp);

    if (!group) {

        goto error;

    }



    QLIST_FOREACH(vbasedev_iter, &group->device_list, next) {

        if (strcmp(vbasedev_iter->name, vdev->vbasedev.name) == 0) {

            error_setg(errp, "device is already attached");

            vfio_put_group(group);

            goto error;

        }

    }



    ret = vfio_get_device(group, vdev->vbasedev.name, &vdev->vbasedev, errp);

    if (ret) {

        vfio_put_group(group);

        goto error;

    }



    vfio_populate_device(vdev, &err);

    if (err) {

        error_propagate(errp, err);

        goto error;

    }



    /* Get a copy of config space */

    ret = pread(vdev->vbasedev.fd, vdev->pdev.config,

                MIN(pci_config_size(&vdev->pdev), vdev->config_size),

                vdev->config_offset);

    if (ret < (int)MIN(pci_config_size(&vdev->pdev), vdev->config_size)) {

        ret = ret < 0 ? -errno : -EFAULT;

        error_setg_errno(errp, -ret, "failed to read device config space");

        goto error;

    }



    /* vfio emulates a lot for us, but some bits need extra love */

    vdev->emulated_config_bits = g_malloc0(vdev->config_size);



    /* QEMU can choose to expose the ROM or not */

    memset(vdev->emulated_config_bits + PCI_ROM_ADDRESS, 0xff, 4);



    /*

     * The PCI spec reserves vendor ID 0xffff as an invalid value.  The

     * device ID is managed by the vendor and need only be a 16-bit value.

     * Allow any 16-bit value for subsystem so they can be hidden or changed.

     */

    if (vdev->vendor_id != PCI_ANY_ID) {

        if (vdev->vendor_id >= 0xffff) {

            error_setg(errp, "invalid PCI vendor ID provided");

            goto error;

        }

        vfio_add_emulated_word(vdev, PCI_VENDOR_ID, vdev->vendor_id, ~0);

        trace_vfio_pci_emulated_vendor_id(vdev->vbasedev.name, vdev->vendor_id);

    } else {

        vdev->vendor_id = pci_get_word(pdev->config + PCI_VENDOR_ID);

    }



    if (vdev->device_id != PCI_ANY_ID) {

        if (vdev->device_id > 0xffff) {

            error_setg(errp, "invalid PCI device ID provided");

            goto error;

        }

        vfio_add_emulated_word(vdev, PCI_DEVICE_ID, vdev->device_id, ~0);

        trace_vfio_pci_emulated_device_id(vdev->vbasedev.name, vdev->device_id);

    } else {

        vdev->device_id = pci_get_word(pdev->config + PCI_DEVICE_ID);

    }



    if (vdev->sub_vendor_id != PCI_ANY_ID) {

        if (vdev->sub_vendor_id > 0xffff) {

            error_setg(errp, "invalid PCI subsystem vendor ID provided");

            goto error;

        }

        vfio_add_emulated_word(vdev, PCI_SUBSYSTEM_VENDOR_ID,

                               vdev->sub_vendor_id, ~0);

        trace_vfio_pci_emulated_sub_vendor_id(vdev->vbasedev.name,

                                              vdev->sub_vendor_id);

    }



    if (vdev->sub_device_id != PCI_ANY_ID) {

        if (vdev->sub_device_id > 0xffff) {

            error_setg(errp, "invalid PCI subsystem device ID provided");

            goto error;

        }

        vfio_add_emulated_word(vdev, PCI_SUBSYSTEM_ID, vdev->sub_device_id, ~0);

        trace_vfio_pci_emulated_sub_device_id(vdev->vbasedev.name,

                                              vdev->sub_device_id);

    }



    /* QEMU can change multi-function devices to single function, or reverse */

    vdev->emulated_config_bits[PCI_HEADER_TYPE] =

                                              PCI_HEADER_TYPE_MULTI_FUNCTION;



    /* Restore or clear multifunction, this is always controlled by QEMU */

    if (vdev->pdev.cap_present & QEMU_PCI_CAP_MULTIFUNCTION) {

        vdev->pdev.config[PCI_HEADER_TYPE] |= PCI_HEADER_TYPE_MULTI_FUNCTION;

    } else {

        vdev->pdev.config[PCI_HEADER_TYPE] &= ~PCI_HEADER_TYPE_MULTI_FUNCTION;

    }



    /*

     * Clear host resource mapping info.  If we choose not to register a

     * BAR, such as might be the case with the option ROM, we can get

     * confusing, unwritable, residual addresses from the host here.

     */

    memset(&vdev->pdev.config[PCI_BASE_ADDRESS_0], 0, 24);

    memset(&vdev->pdev.config[PCI_ROM_ADDRESS], 0, 4);



    vfio_pci_size_rom(vdev);



    vfio_msix_early_setup(vdev, &err);

    if (err) {

        error_propagate(errp, err);

        goto error;

    }



    vfio_bars_setup(vdev);



    ret = vfio_add_capabilities(vdev, errp);

    if (ret) {

        goto out_teardown;

    }



    if (vdev->vga) {

        vfio_vga_quirk_setup(vdev);

    }



    for (i = 0; i < PCI_ROM_SLOT; i++) {

        vfio_bar_quirk_setup(vdev, i);

    }



    if (!vdev->igd_opregion &&

        vdev->features & VFIO_FEATURE_ENABLE_IGD_OPREGION) {

        struct vfio_region_info *opregion;



        if (vdev->pdev.qdev.hotplugged) {

            error_setg(errp,

                       "cannot support IGD OpRegion feature on hotplugged "

                       "device");

            goto out_teardown;

        }



        ret = vfio_get_dev_region_info(&vdev->vbasedev,

                        VFIO_REGION_TYPE_PCI_VENDOR_TYPE | PCI_VENDOR_ID_INTEL,

                        VFIO_REGION_SUBTYPE_INTEL_IGD_OPREGION, &opregion);

        if (ret) {

            error_setg_errno(errp, -ret,

                             "does not support requested IGD OpRegion feature");

            goto out_teardown;

        }



        ret = vfio_pci_igd_opregion_init(vdev, opregion, errp);

        g_free(opregion);

        if (ret) {

            goto out_teardown;

        }

    }



    /* QEMU emulates all of MSI & MSIX */

    if (pdev->cap_present & QEMU_PCI_CAP_MSIX) {

        memset(vdev->emulated_config_bits + pdev->msix_cap, 0xff,

               MSIX_CAP_LENGTH);

    }



    if (pdev->cap_present & QEMU_PCI_CAP_MSI) {

        memset(vdev->emulated_config_bits + pdev->msi_cap, 0xff,

               vdev->msi_cap_size);

    }



    if (vfio_pci_read_config(&vdev->pdev, PCI_INTERRUPT_PIN, 1)) {

        vdev->intx.mmap_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,

                                                  vfio_intx_mmap_enable, vdev);

        pci_device_set_intx_routing_notifier(&vdev->pdev, vfio_intx_update);

        ret = vfio_intx_enable(vdev, errp);

        if (ret) {

            goto out_teardown;

        }

    }



    vfio_register_err_notifier(vdev);

    vfio_register_req_notifier(vdev);

    vfio_setup_resetfn_quirk(vdev);



    return;



out_teardown:

    pci_device_set_intx_routing_notifier(&vdev->pdev, NULL);

    vfio_teardown_msi(vdev);

    vfio_bars_exit(vdev);

error:

    error_prepend(errp, ERR_PREFIX, vdev->vbasedev.name);

}
