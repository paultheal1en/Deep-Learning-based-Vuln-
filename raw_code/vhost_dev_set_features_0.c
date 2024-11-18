static int vhost_dev_set_features(struct vhost_dev *dev, bool enable_log)

{

    uint64_t features = dev->acked_features;

    int r;

    if (enable_log) {

        features |= 0x1ULL << VHOST_F_LOG_ALL;

    }

    r = dev->vhost_ops->vhost_set_features(dev, features);

    if (r < 0) {

        VHOST_OPS_DEBUG("vhost_set_features failed");

    }

    return r < 0 ? -errno : 0;

}
