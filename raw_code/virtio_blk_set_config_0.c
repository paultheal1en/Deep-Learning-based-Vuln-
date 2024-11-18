static void virtio_blk_set_config(VirtIODevice *vdev, const uint8_t *config)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);

    struct virtio_blk_config blkcfg;



    memcpy(&blkcfg, config, sizeof(blkcfg));



    aio_context_acquire(bdrv_get_aio_context(s->bs));

    bdrv_set_enable_write_cache(s->bs, blkcfg.wce != 0);

    aio_context_release(bdrv_get_aio_context(s->bs));

}
