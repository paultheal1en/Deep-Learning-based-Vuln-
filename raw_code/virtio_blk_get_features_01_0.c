static uint32_t virtio_blk_get_features(VirtIODevice *vdev, uint32_t features)

{

    VirtIOBlock *s = to_virtio_blk(vdev);



    features |= (1 << VIRTIO_BLK_F_SEG_MAX);

    features |= (1 << VIRTIO_BLK_F_GEOMETRY);

    features |= (1 << VIRTIO_BLK_F_TOPOLOGY);

    features |= (1 << VIRTIO_BLK_F_BLK_SIZE);

    features |= (1 << VIRTIO_BLK_F_SCSI);



    features |= (1 << VIRTIO_BLK_F_CONFIG_WCE);

    if (bdrv_enable_write_cache(s->bs))

        features |= (1 << VIRTIO_BLK_F_WCE);



    if (bdrv_is_read_only(s->bs))

        features |= 1 << VIRTIO_BLK_F_RO;



    return features;

}
