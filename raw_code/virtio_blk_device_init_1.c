static int virtio_blk_device_init(VirtIODevice *vdev)

{

    DeviceState *qdev = DEVICE(vdev);

    VirtIOBlock *s = VIRTIO_BLK(vdev);

    VirtIOBlkConf *blk = &(s->blk);

    static int virtio_blk_id;



    if (!blk->conf.bs) {

        error_report("drive property not set");

        return -1;

    }

    if (!bdrv_is_inserted(blk->conf.bs)) {

        error_report("Device needs media, but drive is empty");

        return -1;

    }



    blkconf_serial(&blk->conf, &blk->serial);

    if (blkconf_geometry(&blk->conf, NULL, 65535, 255, 255) < 0) {

        return -1;

    }



    virtio_init(vdev, "virtio-blk", VIRTIO_ID_BLOCK,

                sizeof(struct virtio_blk_config));



    vdev->get_config = virtio_blk_update_config;

    vdev->set_config = virtio_blk_set_config;

    vdev->get_features = virtio_blk_get_features;

    vdev->set_status = virtio_blk_set_status;

    vdev->reset = virtio_blk_reset;

    s->bs = blk->conf.bs;

    s->conf = &blk->conf;

    memcpy(&(s->blk), blk, sizeof(struct VirtIOBlkConf));

    s->rq = NULL;

    s->sector_mask = (s->conf->logical_block_size / BDRV_SECTOR_SIZE) - 1;



    s->vq = virtio_add_queue(vdev, 128, virtio_blk_handle_output);

#ifdef CONFIG_VIRTIO_BLK_DATA_PLANE

    if (!virtio_blk_data_plane_create(vdev, blk, &s->dataplane)) {

        virtio_cleanup(vdev);

        return -1;

    }

#endif



    s->change = qemu_add_vm_change_state_handler(virtio_blk_dma_restart_cb, s);

    register_savevm(qdev, "virtio-blk", virtio_blk_id++, 2,

                    virtio_blk_save, virtio_blk_load, s);

    bdrv_set_dev_ops(s->bs, &virtio_block_ops, s);

    bdrv_set_buffer_alignment(s->bs, s->conf->logical_block_size);



    bdrv_iostatus_enable(s->bs);



    add_boot_device_path(s->conf->bootindex, qdev, "/disk@0,0");

    return 0;

}
