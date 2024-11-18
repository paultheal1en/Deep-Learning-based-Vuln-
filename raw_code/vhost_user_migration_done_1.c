static int vhost_user_migration_done(struct vhost_dev *dev, char* mac_addr)

{

    VhostUserMsg msg = { 0 };

    int err;



    assert(dev->vhost_ops->backend_type == VHOST_BACKEND_TYPE_USER);



    /* If guest supports GUEST_ANNOUNCE do nothing */

    if (virtio_has_feature(dev->acked_features, VIRTIO_NET_F_GUEST_ANNOUNCE)) {

        return 0;

    }



    /* if backend supports VHOST_USER_PROTOCOL_F_RARP ask it to send the RARP */

    if (virtio_has_feature(dev->protocol_features,

                           VHOST_USER_PROTOCOL_F_RARP)) {

        msg.request = VHOST_USER_SEND_RARP;

        msg.flags = VHOST_USER_VERSION;

        memcpy((char *)&msg.u64, mac_addr, 6);

        msg.size = sizeof(m.u64);



        err = vhost_user_write(dev, &msg, NULL, 0);

        return err;

    }

    return -1;

}
