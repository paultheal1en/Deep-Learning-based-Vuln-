static int virtio_ccw_device_init(VirtioCcwDevice *dev, VirtIODevice *vdev)

{

    unsigned int cssid = 0;

    unsigned int ssid = 0;

    unsigned int schid;

    unsigned int devno;

    bool have_devno = false;

    bool found = false;

    SubchDev *sch;

    int ret;

    int num;

    DeviceState *parent = DEVICE(dev);



    sch = g_malloc0(sizeof(SubchDev));



    sch->driver_data = dev;

    dev->sch = sch;



    dev->indicators = NULL;



    /* Initialize subchannel structure. */

    sch->channel_prog = 0x0;

    sch->last_cmd_valid = false;

    sch->thinint_active = false;

    /*

     * Use a device number if provided. Otherwise, fall back to subchannel

     * number.

     */

    if (dev->bus_id) {

        num = sscanf(dev->bus_id, "%x.%x.%04x", &cssid, &ssid, &devno);

        if (num == 3) {

            if ((cssid > MAX_CSSID) || (ssid > MAX_SSID)) {

                ret = -EINVAL;

                error_report("Invalid cssid or ssid: cssid %x, ssid %x",

                             cssid, ssid);

                goto out_err;

            }

            /* Enforce use of virtual cssid. */

            if (cssid != VIRTUAL_CSSID) {

                ret = -EINVAL;

                error_report("cssid %x not valid for virtio devices", cssid);

                goto out_err;

            }

            if (css_devno_used(cssid, ssid, devno)) {

                ret = -EEXIST;

                error_report("Device %x.%x.%04x already exists", cssid, ssid,

                             devno);

                goto out_err;

            }

            sch->cssid = cssid;

            sch->ssid = ssid;

            sch->devno = devno;

            have_devno = true;

        } else {

            ret = -EINVAL;

            error_report("Malformed devno parameter '%s'", dev->bus_id);

            goto out_err;

        }

    }



    /* Find the next free id. */

    if (have_devno) {

        for (schid = 0; schid <= MAX_SCHID; schid++) {

            if (!css_find_subch(1, cssid, ssid, schid)) {

                sch->schid = schid;

                css_subch_assign(cssid, ssid, schid, devno, sch);

                found = true;

                break;

            }

        }

        if (!found) {

            ret = -ENODEV;

            error_report("No free subchannel found for %x.%x.%04x", cssid, ssid,

                         devno);

            goto out_err;

        }

        trace_virtio_ccw_new_device(cssid, ssid, schid, devno,

                                    "user-configured");

    } else {

        cssid = VIRTUAL_CSSID;

        for (ssid = 0; ssid <= MAX_SSID; ssid++) {

            for (schid = 0; schid <= MAX_SCHID; schid++) {

                if (!css_find_subch(1, cssid, ssid, schid)) {

                    sch->cssid = cssid;

                    sch->ssid = ssid;

                    sch->schid = schid;

                    devno = schid;

                    /*

                     * If the devno is already taken, look further in this

                     * subchannel set.

                     */

                    while (css_devno_used(cssid, ssid, devno)) {

                        if (devno == MAX_SCHID) {

                            devno = 0;

                        } else if (devno == schid - 1) {

                            ret = -ENODEV;

                            error_report("No free devno found");

                            goto out_err;

                        } else {

                            devno++;

                        }

                    }

                    sch->devno = devno;

                    css_subch_assign(cssid, ssid, schid, devno, sch);

                    found = true;

                    break;

                }

            }

            if (found) {

                break;

            }

        }

        if (!found) {

            ret = -ENODEV;

            error_report("Virtual channel subsystem is full!");

            goto out_err;

        }

        trace_virtio_ccw_new_device(cssid, ssid, schid, devno,

                                    "auto-configured");

    }



    /* Build initial schib. */

    css_sch_build_virtual_schib(sch, 0, VIRTIO_CCW_CHPID_TYPE);



    sch->ccw_cb = virtio_ccw_cb;



    /* Build senseid data. */

    memset(&sch->id, 0, sizeof(SenseId));

    sch->id.reserved = 0xff;

    sch->id.cu_type = VIRTIO_CCW_CU_TYPE;

    sch->id.cu_model = vdev->device_id;



    /* Only the first 32 feature bits are used. */

    dev->host_features[0] = virtio_bus_get_vdev_features(&dev->bus,

                                                         dev->host_features[0]);



    dev->host_features[0] |= 0x1 << VIRTIO_F_NOTIFY_ON_EMPTY;

    dev->host_features[0] |= 0x1 << VIRTIO_F_BAD_FEATURE;



    css_generate_sch_crws(sch->cssid, sch->ssid, sch->schid,

                          parent->hotplugged, 1);

    return 0;



out_err:

    dev->sch = NULL;

    g_free(sch);

    return ret;

}
