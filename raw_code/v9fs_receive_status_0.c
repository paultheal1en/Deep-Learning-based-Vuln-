static int v9fs_receive_status(V9fsProxy *proxy,

                               struct iovec *reply, int *status)

{

    int retval;

    ProxyHeader header;



    *status = 0;

    reply->iov_len = 0;

    retval = socket_read(proxy->sockfd, reply->iov_base, PROXY_HDR_SZ);

    if (retval < 0) {

        return retval;

    }

    reply->iov_len = PROXY_HDR_SZ;

    proxy_unmarshal(reply, 0, "dd", &header.type, &header.size);

    if (header.size != sizeof(int)) {

        *status = -ENOBUFS;

        return 0;

    }

    retval = socket_read(proxy->sockfd,

                         reply->iov_base + PROXY_HDR_SZ, header.size);

    if (retval < 0) {

        return retval;

    }

    reply->iov_len += header.size;

    proxy_unmarshal(reply, PROXY_HDR_SZ, "d", status);

    return 0;

}
