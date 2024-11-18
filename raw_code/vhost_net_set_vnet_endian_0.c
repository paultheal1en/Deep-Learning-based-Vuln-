static int vhost_net_set_vnet_endian(VirtIODevice *dev, NetClientState *peer,

                                     bool set)

{

    int r = 0;



    if (virtio_has_feature(dev, VIRTIO_F_VERSION_1) ||

        (virtio_legacy_is_cross_endian(dev) && !virtio_is_big_endian(dev))) {

        r = qemu_set_vnet_le(peer, set);

        if (r) {

            error_report("backend does not support LE vnet headers");

        }

    } else if (virtio_legacy_is_cross_endian(dev)) {

        r = qemu_set_vnet_be(peer, set);

        if (r) {

            error_report("backend does not support BE vnet headers");

        }

    }



    return r;

}
