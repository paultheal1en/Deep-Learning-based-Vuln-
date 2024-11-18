static int vfio_pci_hot_reset(VFIOPCIDevice *vdev, bool single)

{

    VFIOGroup *group;

    struct vfio_pci_hot_reset_info *info;

    struct vfio_pci_dependent_device *devices;

    struct vfio_pci_hot_reset *reset;

    int32_t *fds;

    int ret, i, count;

    bool multi = false;



    trace_vfio_pci_hot_reset(vdev->vbasedev.name, single ? "one" : "multi");



    vfio_pci_pre_reset(vdev);

    vdev->vbasedev.needs_reset = false;



    info = g_malloc0(sizeof(*info));

    info->argsz = sizeof(*info);



    ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_GET_PCI_HOT_RESET_INFO, info);

    if (ret && errno != ENOSPC) {

        ret = -errno;

        if (!vdev->has_pm_reset) {

            error_report("vfio: Cannot reset device %04x:%02x:%02x.%x, "

                         "no available reset mechanism.", vdev->host.domain,

                         vdev->host.bus, vdev->host.slot, vdev->host.function);

        }

        goto out_single;

    }



    count = info->count;

    info = g_realloc(info, sizeof(*info) + (count * sizeof(*devices)));

    info->argsz = sizeof(*info) + (count * sizeof(*devices));

    devices = &info->devices[0];



    ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_GET_PCI_HOT_RESET_INFO, info);

    if (ret) {

        ret = -errno;

        error_report("vfio: hot reset info failed: %m");

        goto out_single;

    }



    trace_vfio_pci_hot_reset_has_dep_devices(vdev->vbasedev.name);



    /* Verify that we have all the groups required */

    for (i = 0; i < info->count; i++) {

        PCIHostDeviceAddress host;

        VFIOPCIDevice *tmp;

        VFIODevice *vbasedev_iter;



        host.domain = devices[i].segment;

        host.bus = devices[i].bus;

        host.slot = PCI_SLOT(devices[i].devfn);

        host.function = PCI_FUNC(devices[i].devfn);



        trace_vfio_pci_hot_reset_dep_devices(host.domain,

                host.bus, host.slot, host.function, devices[i].group_id);



        if (vfio_pci_host_match(&host, &vdev->host)) {

            continue;

        }



        QLIST_FOREACH(group, &vfio_group_list, next) {

            if (group->groupid == devices[i].group_id) {

                break;

            }

        }



        if (!group) {

            if (!vdev->has_pm_reset) {

                error_report("vfio: Cannot reset device %s, "

                             "depends on group %d which is not owned.",

                             vdev->vbasedev.name, devices[i].group_id);

            }

            ret = -EPERM;

            goto out;

        }



        /* Prep dependent devices for reset and clear our marker. */

        QLIST_FOREACH(vbasedev_iter, &group->device_list, next) {

            if (vbasedev_iter->type != VFIO_DEVICE_TYPE_PCI) {

                continue;

            }

            tmp = container_of(vbasedev_iter, VFIOPCIDevice, vbasedev);

            if (vfio_pci_host_match(&host, &tmp->host)) {

                if (single) {

                    ret = -EINVAL;

                    goto out_single;

                }

                vfio_pci_pre_reset(tmp);

                tmp->vbasedev.needs_reset = false;

                multi = true;

                break;

            }

        }

    }



    if (!single && !multi) {

        ret = -EINVAL;

        goto out_single;

    }



    /* Determine how many group fds need to be passed */

    count = 0;

    QLIST_FOREACH(group, &vfio_group_list, next) {

        for (i = 0; i < info->count; i++) {

            if (group->groupid == devices[i].group_id) {

                count++;

                break;

            }

        }

    }



    reset = g_malloc0(sizeof(*reset) + (count * sizeof(*fds)));

    reset->argsz = sizeof(*reset) + (count * sizeof(*fds));

    fds = &reset->group_fds[0];



    /* Fill in group fds */

    QLIST_FOREACH(group, &vfio_group_list, next) {

        for (i = 0; i < info->count; i++) {

            if (group->groupid == devices[i].group_id) {

                fds[reset->count++] = group->fd;

                break;

            }

        }

    }



    /* Bus reset! */

    ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_PCI_HOT_RESET, reset);

    g_free(reset);



    trace_vfio_pci_hot_reset_result(vdev->vbasedev.name,

                                    ret ? "%m" : "Success");



out:

    /* Re-enable INTx on affected devices */

    for (i = 0; i < info->count; i++) {

        PCIHostDeviceAddress host;

        VFIOPCIDevice *tmp;

        VFIODevice *vbasedev_iter;



        host.domain = devices[i].segment;

        host.bus = devices[i].bus;

        host.slot = PCI_SLOT(devices[i].devfn);

        host.function = PCI_FUNC(devices[i].devfn);



        if (vfio_pci_host_match(&host, &vdev->host)) {

            continue;

        }



        QLIST_FOREACH(group, &vfio_group_list, next) {

            if (group->groupid == devices[i].group_id) {

                break;

            }

        }



        if (!group) {

            break;

        }



        QLIST_FOREACH(vbasedev_iter, &group->device_list, next) {

            if (vbasedev_iter->type != VFIO_DEVICE_TYPE_PCI) {

                continue;

            }

            tmp = container_of(vbasedev_iter, VFIOPCIDevice, vbasedev);

            if (vfio_pci_host_match(&host, &tmp->host)) {

                vfio_pci_post_reset(tmp);

                break;

            }

        }

    }

out_single:

    vfio_pci_post_reset(vdev);

    g_free(info);



    return ret;

}
