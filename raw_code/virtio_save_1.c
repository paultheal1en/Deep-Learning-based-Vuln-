void virtio_save(VirtIODevice *vdev, QEMUFile *f)

{

    BusState *qbus = qdev_get_parent_bus(DEVICE(vdev));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_GET_CLASS(vdev);

    uint32_t guest_features_lo = (vdev->guest_features & 0xffffffff);

    int i;



    if (k->save_config) {

        k->save_config(qbus->parent, f);

    }



    qemu_put_8s(f, &vdev->status);

    qemu_put_8s(f, &vdev->isr);

    qemu_put_be16s(f, &vdev->queue_sel);

    qemu_put_be32s(f, &guest_features_lo);

    qemu_put_be32(f, vdev->config_len);

    qemu_put_buffer(f, vdev->config, vdev->config_len);



    for (i = 0; i < VIRTIO_QUEUE_MAX; i++) {

        if (vdev->vq[i].vring.num == 0)

            break;

    }



    qemu_put_be32(f, i);



    for (i = 0; i < VIRTIO_QUEUE_MAX; i++) {

        if (vdev->vq[i].vring.num == 0)

            break;



        qemu_put_be32(f, vdev->vq[i].vring.num);

        if (k->has_variable_vring_alignment) {

            qemu_put_be32(f, vdev->vq[i].vring.align);

        }

        /* XXX virtio-1 devices */

        qemu_put_be64(f, vdev->vq[i].vring.desc);

        qemu_put_be16s(f, &vdev->vq[i].last_avail_idx);

        if (k->save_queue) {

            k->save_queue(qbus->parent, i, f);

        }

    }



    if (vdc->save != NULL) {

        vdc->save(vdev, f);

    }



    if (vdc->vmsd) {

        vmstate_save_state(f, vdc->vmsd, vdev, NULL);

    }



    /* Subsections */

    vmstate_save_state(f, &vmstate_virtio, vdev, NULL);

}
