int vhost_dev_start(struct vhost_dev *hdev, VirtIODevice *vdev)

{

    int i, r;



    hdev->started = true;



    r = vhost_dev_set_features(hdev, hdev->log_enabled);

    if (r < 0) {

        goto fail_features;

    }

    r = hdev->vhost_ops->vhost_call(hdev, VHOST_SET_MEM_TABLE, hdev->mem);

    if (r < 0) {

        r = -errno;

        goto fail_mem;

    }

    for (i = 0; i < hdev->nvqs; ++i) {

        r = vhost_virtqueue_start(hdev,

                                  vdev,

                                  hdev->vqs + i,

                                  hdev->vq_index + i);

        if (r < 0) {

            goto fail_vq;

        }

    }



    if (hdev->log_enabled) {

        hdev->log_size = vhost_get_log_size(hdev);

        hdev->log = hdev->log_size ?

            g_malloc0(hdev->log_size * sizeof *hdev->log) : NULL;

        r = hdev->vhost_ops->vhost_call(hdev, VHOST_SET_LOG_BASE, hdev->log);

        if (r < 0) {

            r = -errno;

            goto fail_log;

        }

    }



    return 0;

fail_log:

fail_vq:

    while (--i >= 0) {

        vhost_virtqueue_stop(hdev,

                             vdev,

                             hdev->vqs + i,

                             hdev->vq_index + i);

    }

    i = hdev->nvqs;

fail_mem:

fail_features:



    hdev->started = false;

    return r;

}
