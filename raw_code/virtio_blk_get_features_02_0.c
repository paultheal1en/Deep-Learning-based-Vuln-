static uint32_t virtio_blk_get_features(VirtIODevice *vdev)

{

    VirtIOBlock *s = to_virtio_blk(vdev);

    uint32_t features = 0;



    features |= (1 << VIRTIO_BLK_F_SEG_MAX);

    features |= (1 << VIRTIO_BLK_F_GEOMETRY);



    if (bdrv_enable_write_cache(s->bs))

        features |= (1 << VIRTIO_BLK_F_WCACHE);

#ifdef __linux__

    features |= (1 << VIRTIO_BLK_F_SCSI);

#endif

    if (strcmp(s->serial_str, "0"))

        features |= 1 << VIRTIO_BLK_F_IDENTIFY;

    

    if (bdrv_is_read_only(s->bs))

        features |= 1 << VIRTIO_BLK_F_RO;



    return features;

}
