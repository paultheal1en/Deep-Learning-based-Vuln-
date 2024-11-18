static void virtio_blk_device_realize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VirtIOBlock *s = VIRTIO_BLK(dev);

    VirtIOBlkConf *conf = &s->conf;

    Error *err = NULL;

    static int virtio_blk_id;



    if (!conf->conf.bs) {

        error_setg(errp, "drive property not set");

        return;

    }

    if (!bdrv_is_inserted(conf->conf.bs)) {

        error_setg(errp, "Device needs media, but drive is empty");

        return;

    }



    blkconf_serial(&conf->conf, &conf->serial);

    s->original_wce = bdrv_enable_write_cache(conf->conf.bs);

    blkconf_geometry(&conf->conf, NULL, 65535, 255, 255, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    virtio_init(vdev, "virtio-blk", VIRTIO_ID_BLOCK,

                sizeof(struct virtio_blk_config));



    s->bs = conf->conf.bs;

    s->rq = NULL;

    s->sector_mask = (s->conf.conf.logical_block_size / BDRV_SECTOR_SIZE) - 1;



    s->vq = virtio_add_queue(vdev, 128, virtio_blk_handle_output);

    s->complete_request = virtio_blk_complete_request;

    virtio_blk_data_plane_create(vdev, conf, &s->dataplane, &err);

    if (err != NULL) {

        error_propagate(errp, err);

        virtio_cleanup(vdev);

        return;

    }

    s->migration_state_notifier.notify = virtio_blk_migration_state_changed;

    add_migration_state_change_notifier(&s->migration_state_notifier);



    s->change = qemu_add_vm_change_state_handler(virtio_blk_dma_restart_cb, s);

    register_savevm(dev, "virtio-blk", virtio_blk_id++, 2,

                    virtio_blk_save, virtio_blk_load, s);

    bdrv_set_dev_ops(s->bs, &virtio_block_ops, s);

    bdrv_set_guest_block_size(s->bs, s->conf.conf.logical_block_size);



    bdrv_iostatus_enable(s->bs);

}
