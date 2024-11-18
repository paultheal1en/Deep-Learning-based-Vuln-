static void virtio_blk_update_config(VirtIODevice *vdev, uint8_t *config)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);

    BlockConf *conf = &s->conf.conf;

    struct virtio_blk_config blkcfg;

    uint64_t capacity;

    int blk_size = conf->logical_block_size;



    bdrv_get_geometry(s->bs, &capacity);

    memset(&blkcfg, 0, sizeof(blkcfg));

    virtio_stq_p(vdev, &blkcfg.capacity, capacity);

    virtio_stl_p(vdev, &blkcfg.seg_max, 128 - 2);

    virtio_stw_p(vdev, &blkcfg.cylinders, conf->cyls);

    virtio_stl_p(vdev, &blkcfg.blk_size, blk_size);

    virtio_stw_p(vdev, &blkcfg.min_io_size, conf->min_io_size / blk_size);

    virtio_stw_p(vdev, &blkcfg.opt_io_size, conf->opt_io_size / blk_size);

    blkcfg.heads = conf->heads;

    /*

     * We must ensure that the block device capacity is a multiple of

     * the logical block size. If that is not the case, let's use

     * sector_mask to adopt the geometry to have a correct picture.

     * For those devices where the capacity is ok for the given geometry

     * we don't touch the sector value of the geometry, since some devices

     * (like s390 dasd) need a specific value. Here the capacity is already

     * cyls*heads*secs*blk_size and the sector value is not block size

     * divided by 512 - instead it is the amount of blk_size blocks

     * per track (cylinder).

     */

    if (bdrv_getlength(s->bs) /  conf->heads / conf->secs % blk_size) {

        blkcfg.sectors = conf->secs & ~s->sector_mask;

    } else {

        blkcfg.sectors = conf->secs;

    }

    blkcfg.size_max = 0;

    blkcfg.physical_block_exp = get_physical_block_exp(conf);

    blkcfg.alignment_offset = 0;

    blkcfg.wce = bdrv_enable_write_cache(s->bs);

    memcpy(config, &blkcfg, sizeof(struct virtio_blk_config));

}
