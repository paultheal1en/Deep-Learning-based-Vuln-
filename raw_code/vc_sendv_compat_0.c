static ssize_t vc_sendv_compat(VLANClientState *vc, const struct iovec *iov,

                               int iovcnt)

{

    uint8_t buffer[4096];

    size_t offset = 0;

    int i;



    for (i = 0; i < iovcnt; i++) {

        size_t len;



        len = MIN(sizeof(buffer) - offset, iov[i].iov_len);

        memcpy(buffer + offset, iov[i].iov_base, len);

        offset += len;

    }



    vc->receive(vc->opaque, buffer, offset);



    return offset;

}
