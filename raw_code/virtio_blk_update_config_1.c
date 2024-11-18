static void virtio_blk_update_config(VirtIODevice *vdev, uint8_t *config)

{

    VirtIOBlock *s = to_virtio_blk(vdev);

    struct virtio_blk_config blkcfg;

    uint64_t capacity;

    int cylinders, heads, secs;



    bdrv_get_geometry(s->bs, &capacity);

    bdrv_get_geometry_hint(s->bs, &cylinders, &heads, &secs);

    stq_raw(&blkcfg.capacity, capacity);

    stl_raw(&blkcfg.seg_max, 128 - 2);

    stw_raw(&blkcfg.cylinders, cylinders);

    blkcfg.heads = heads;

    blkcfg.sectors = secs;


    memcpy(config, &blkcfg, sizeof(blkcfg));

}