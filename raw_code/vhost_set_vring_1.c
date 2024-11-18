static int vhost_set_vring(struct vhost_dev *dev,

                           unsigned long int request,

                           struct vhost_vring_state *ring)

{

    VhostUserMsg msg = {

        .request = request,

        .flags = VHOST_USER_VERSION,

        .state = *ring,

        .size = sizeof(*ring),

    };



    vhost_user_write(dev, &msg, NULL, 0);



    return 0;

}
