void vhost_dev_stop(struct vhost_dev *hdev, VirtIODevice *vdev)

{

    int i, r;



    for (i = 0; i < hdev->nvqs; ++i) {

        vhost_virtqueue_cleanup(hdev,

                                vdev,

                                hdev->vqs + i,

                                i);

    }

    vhost_client_sync_dirty_bitmap(&hdev->client, 0,

                                   (target_phys_addr_t)~0x0ull);

    r = vdev->binding->set_guest_notifiers(vdev->binding_opaque, false);

    if (r < 0) {

        fprintf(stderr, "vhost guest notifier cleanup failed: %d\n", r);

        fflush(stderr);

    }

    assert (r >= 0);



    hdev->started = false;

    g_free(hdev->log);

    hdev->log = NULL;

    hdev->log_size = 0;

}
