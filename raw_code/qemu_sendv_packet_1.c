ssize_t qemu_sendv_packet(VLANClientState *vc1, const struct iovec *iov,

                          int iovcnt)

{

    VLANState *vlan = vc1->vlan;

    VLANClientState *vc;

    ssize_t max_len = 0;



    if (vc1->link_down)

        return calc_iov_length(iov, iovcnt);



    for (vc = vlan->first_client; vc != NULL; vc = vc->next) {

        ssize_t len = 0;



        if (vc == vc1)

            continue;



        if (vc->link_down)

            len = calc_iov_length(iov, iovcnt);

        else if (vc->fd_readv)

            len = vc->fd_readv(vc->opaque, iov, iovcnt);

        else if (vc->fd_read)

            len = vc_sendv_compat(vc, iov, iovcnt);



        max_len = MAX(max_len, len);

    }



    return max_len;

}
