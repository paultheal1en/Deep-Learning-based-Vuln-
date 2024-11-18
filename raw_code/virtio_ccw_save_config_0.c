static void virtio_ccw_save_config(DeviceState *d, QEMUFile *f)

{

    VirtioCcwDevice *dev = VIRTIO_CCW_DEVICE(d);

    CcwDevice *ccw_dev = CCW_DEVICE(d);

    SubchDev *s = ccw_dev->sch;

    VirtIODevice *vdev = virtio_ccw_get_vdev(s);



    subch_device_save(s, f);

    if (dev->indicators != NULL) {

        qemu_put_be32(f, dev->indicators->len);

        qemu_put_be64(f, dev->indicators->addr);

    } else {

        qemu_put_be32(f, 0);

        qemu_put_be64(f, 0UL);

    }

    if (dev->indicators2 != NULL) {

        qemu_put_be32(f, dev->indicators2->len);

        qemu_put_be64(f, dev->indicators2->addr);

    } else {

        qemu_put_be32(f, 0);

        qemu_put_be64(f, 0UL);

    }

    if (dev->summary_indicator != NULL) {

        qemu_put_be32(f, dev->summary_indicator->len);

        qemu_put_be64(f, dev->summary_indicator->addr);

    } else {

        qemu_put_be32(f, 0);

        qemu_put_be64(f, 0UL);

    }

    qemu_put_be16(f, vdev->config_vector);

    qemu_put_be64(f, dev->routes.adapter.ind_offset);

    qemu_put_byte(f, dev->thinint_isc);

    qemu_put_be32(f, dev->revision);

}
