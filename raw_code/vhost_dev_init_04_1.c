int vhost_dev_init(struct vhost_dev *hdev, int devfd, bool force)

{

    uint64_t features;

    int r;

    if (devfd >= 0) {

        hdev->control = devfd;

    } else {

        hdev->control = open("/dev/vhost-net", O_RDWR);

        if (hdev->control < 0) {

            return -errno;

        }

    }

    r = ioctl(hdev->control, VHOST_SET_OWNER, NULL);

    if (r < 0) {

        goto fail;

    }



    r = ioctl(hdev->control, VHOST_GET_FEATURES, &features);

    if (r < 0) {

        goto fail;

    }

    hdev->features = features;



    hdev->client.set_memory = vhost_client_set_memory;

    hdev->client.sync_dirty_bitmap = vhost_client_sync_dirty_bitmap;

    hdev->client.migration_log = vhost_client_migration_log;

    hdev->client.log_start = NULL;

    hdev->client.log_stop = NULL;

    hdev->mem = g_malloc0(offsetof(struct vhost_memory, regions));

    hdev->log = NULL;

    hdev->log_size = 0;

    hdev->log_enabled = false;

    hdev->started = false;

    cpu_register_phys_memory_client(&hdev->client);

    hdev->force = force;

    return 0;

fail:

    r = -errno;

    close(hdev->control);

    return r;

}
