static uint64_t virtio_blk_get_features(VirtIODevice *vdev, uint64_t features,

                                        Error **errp)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);



    virtio_add_feature(&features, VIRTIO_BLK_F_SEG_MAX);

    virtio_add_feature(&features, VIRTIO_BLK_F_GEOMETRY);

    virtio_add_feature(&features, VIRTIO_BLK_F_TOPOLOGY);

    virtio_add_feature(&features, VIRTIO_BLK_F_BLK_SIZE);

    if (__virtio_has_feature(features, VIRTIO_F_VERSION_1)) {

        if (s->conf.scsi) {

            error_setg(errp, "Please set scsi=off for virtio-blk devices in order to use virtio 1.0");

            return 0;

        }

    } else {

        virtio_clear_feature(&features, VIRTIO_F_ANY_LAYOUT);

        virtio_add_feature(&features, VIRTIO_BLK_F_SCSI);

    }



    if (s->conf.config_wce) {

        virtio_add_feature(&features, VIRTIO_BLK_F_CONFIG_WCE);

    }

    if (blk_enable_write_cache(s->blk)) {

        virtio_add_feature(&features, VIRTIO_BLK_F_WCE);

    }

    if (blk_is_read_only(s->blk)) {

        virtio_add_feature(&features, VIRTIO_BLK_F_RO);

    }



    return features;

}
