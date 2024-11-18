int virtio_load(VirtIODevice *vdev, QEMUFile *f)

{

    int num, i, ret;

    uint32_t features;

    uint32_t supported_features =

        vdev->binding->get_features(vdev->binding_opaque);




    if (vdev->binding->load_config) {

        ret = vdev->binding->load_config(vdev->binding_opaque, f);

        if (ret)

            return ret;

    }



    qemu_get_8s(f, &vdev->status);

    qemu_get_8s(f, &vdev->isr);

    qemu_get_be16s(f, &vdev->queue_sel);

    qemu_get_be32s(f, &features);

    if (features & ~supported_features) {

        fprintf(stderr, "Features 0x%x unsupported. Allowed features: 0x%x\n",

                features, supported_features);

        return -1;

    }

    if (vdev->set_features)

        vdev->set_features(vdev, features);

    vdev->guest_features = features;

    vdev->config_len = qemu_get_be32(f);

    qemu_get_buffer(f, vdev->config, vdev->config_len);



    num = qemu_get_be32(f);



    for (i = 0; i < num; i++) {

        vdev->vq[i].vring.num = qemu_get_be32(f);

        vdev->vq[i].pa = qemu_get_be64(f);

        qemu_get_be16s(f, &vdev->vq[i].last_avail_idx);



        if (vdev->vq[i].pa) {

            virtqueue_init(&vdev->vq[i]);

        }

	num_heads = vring_avail_idx(&vdev->vq[i]) - vdev->vq[i].last_avail_idx;

	/* Check it isn't doing very strange things with descriptor numbers. */

	if (num_heads > vdev->vq[i].vring.num) {

		fprintf(stderr, "VQ %d size 0x%x Guest index 0x%x "

                        "inconsistent with Host index 0x%x: delta 0x%x\n",

			i, vdev->vq[i].vring.num,

                        vring_avail_idx(&vdev->vq[i]),

                        vdev->vq[i].last_avail_idx, num_heads);

		return -1;

	}

        if (vdev->binding->load_queue) {

            ret = vdev->binding->load_queue(vdev->binding_opaque, i, f);

            if (ret)

                return ret;

        }

    }



    virtio_notify_vector(vdev, VIRTIO_NO_VECTOR);

    return 0;

}