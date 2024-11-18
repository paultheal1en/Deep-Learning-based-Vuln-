VirtIODevice *virtio_blk_init(DeviceState *dev, BlockConf *conf)

{

    VirtIOBlock *s;

    int cylinders, heads, secs;

    static int virtio_blk_id;

    DriveInfo *dinfo;



    if (!conf->bs) {

        error_report("virtio-blk-pci: drive property not set");









    s = (VirtIOBlock *)virtio_common_init("virtio-blk", VIRTIO_ID_BLOCK,

                                          sizeof(struct virtio_blk_config),

                                          sizeof(VirtIOBlock));



    s->vdev.get_config = virtio_blk_update_config;

    s->vdev.get_features = virtio_blk_get_features;

    s->vdev.reset = virtio_blk_reset;

    s->bs = conf->bs;

    s->conf = conf;

    s->rq = NULL;

    s->sector_mask = (s->conf->logical_block_size / BDRV_SECTOR_SIZE) - 1;

    bdrv_guess_geometry(s->bs, &cylinders, &heads, &secs);



    /* NB: per existing s/n string convention the string is terminated

     * by '\0' only when less than sizeof (s->sn)

     */

    dinfo = drive_get_by_blockdev(s->bs);

    strncpy(s->sn, dinfo->serial, sizeof (s->sn));



    s->vq = virtio_add_queue(&s->vdev, 128, virtio_blk_handle_output);



    qemu_add_vm_change_state_handler(virtio_blk_dma_restart_cb, s);

    register_savevm(dev, "virtio-blk", virtio_blk_id++, 2,

                    virtio_blk_save, virtio_blk_load, s);

    bdrv_set_removable(s->bs, 0);



    return &s->vdev;
