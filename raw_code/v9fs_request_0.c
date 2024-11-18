static int v9fs_request(V9fsProxy *proxy, int type,

                        void *response, const char *fmt, ...)

{

    dev_t rdev;

    va_list ap;

    int size = 0;

    int retval = 0;

    uint64_t offset;

    ProxyHeader header = { 0, 0};

    struct timespec spec[2];

    int flags, mode, uid, gid;

    V9fsString *name, *value;

    V9fsString *path, *oldpath;

    struct iovec *iovec = NULL, *reply = NULL;



    qemu_mutex_lock(&proxy->mutex);



    if (proxy->sockfd == -1) {

        retval = -EIO;

        goto err_out;

    }

    iovec = &proxy->out_iovec;

    reply = &proxy->in_iovec;

    va_start(ap, fmt);

    switch (type) {

    case T_OPEN:

        path = va_arg(ap, V9fsString *);

        flags = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "sd", path, flags);

        if (retval > 0) {

            header.size = retval;

            header.type = T_OPEN;

        }

        break;

    case T_CREATE:

        path = va_arg(ap, V9fsString *);

        flags = va_arg(ap, int);

        mode = va_arg(ap, int);

        uid = va_arg(ap, int);

        gid = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "sdddd", path,

                                    flags, mode, uid, gid);

        if (retval > 0) {

            header.size = retval;

            header.type = T_CREATE;

        }

        break;

    case T_MKNOD:

        path = va_arg(ap, V9fsString *);

        mode = va_arg(ap, int);

        rdev = va_arg(ap, long int);

        uid = va_arg(ap, int);

        gid = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "ddsdq",

                                    uid, gid, path, mode, rdev);

        if (retval > 0) {

            header.size = retval;

            header.type = T_MKNOD;

        }

        break;

    case T_MKDIR:

        path = va_arg(ap, V9fsString *);

        mode = va_arg(ap, int);

        uid = va_arg(ap, int);

        gid = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "ddsd",

                                    uid, gid, path, mode);

        if (retval > 0) {

            header.size = retval;

            header.type = T_MKDIR;

        }

        break;

    case T_SYMLINK:

        oldpath = va_arg(ap, V9fsString *);

        path = va_arg(ap, V9fsString *);

        uid = va_arg(ap, int);

        gid = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "ddss",

                                    uid, gid, oldpath, path);

        if (retval > 0) {

            header.size = retval;

            header.type = T_SYMLINK;

        }

        break;

    case T_LINK:

        oldpath = va_arg(ap, V9fsString *);

        path = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "ss",

                                    oldpath, path);

        if (retval > 0) {

            header.size = retval;

            header.type = T_LINK;

        }

        break;

    case T_LSTAT:

        path = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "s", path);

        if (retval > 0) {

            header.size = retval;

            header.type = T_LSTAT;

        }

        break;

    case T_READLINK:

        path = va_arg(ap, V9fsString *);

        size = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "sd", path, size);

        if (retval > 0) {

            header.size = retval;

            header.type = T_READLINK;

        }

        break;

    case T_STATFS:

        path = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "s", path);

        if (retval > 0) {

            header.size = retval;

            header.type = T_STATFS;

        }

        break;

    case T_CHMOD:

        path = va_arg(ap, V9fsString *);

        mode = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "sd", path, mode);

        if (retval > 0) {

            header.size = retval;

            header.type = T_CHMOD;

        }

        break;

    case T_CHOWN:

        path = va_arg(ap, V9fsString *);

        uid = va_arg(ap, int);

        gid = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "sdd", path, uid, gid);

        if (retval > 0) {

            header.size = retval;

            header.type = T_CHOWN;

        }

        break;

    case T_TRUNCATE:

        path = va_arg(ap, V9fsString *);

        offset = va_arg(ap, uint64_t);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "sq", path, offset);

        if (retval > 0) {

            header.size = retval;

            header.type = T_TRUNCATE;

        }

        break;

    case T_UTIME:

        path = va_arg(ap, V9fsString *);

        spec[0].tv_sec = va_arg(ap, long);

        spec[0].tv_nsec = va_arg(ap, long);

        spec[1].tv_sec = va_arg(ap, long);

        spec[1].tv_nsec = va_arg(ap, long);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "sqqqq", path,

                                    spec[0].tv_sec, spec[1].tv_nsec,

                                    spec[1].tv_sec, spec[1].tv_nsec);

        if (retval > 0) {

            header.size = retval;

            header.type = T_UTIME;

        }

        break;

    case T_RENAME:

        oldpath = va_arg(ap, V9fsString *);

        path = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "ss", oldpath, path);

        if (retval > 0) {

            header.size = retval;

            header.type = T_RENAME;

        }

        break;

    case T_REMOVE:

        path = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "s", path);

        if (retval > 0) {

            header.size = retval;

            header.type = T_REMOVE;

        }

        break;

    case T_LGETXATTR:

        size = va_arg(ap, int);

        path = va_arg(ap, V9fsString *);

        name = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ,

                                    "dss", size, path, name);

        if (retval > 0) {

            header.size = retval;

            header.type = T_LGETXATTR;

        }

        break;

    case T_LLISTXATTR:

        size = va_arg(ap, int);

        path = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "ds", size, path);

        if (retval > 0) {

            header.size = retval;

            header.type = T_LLISTXATTR;

        }

        break;

    case T_LSETXATTR:

        path = va_arg(ap, V9fsString *);

        name = va_arg(ap, V9fsString *);

        value = va_arg(ap, V9fsString *);

        size = va_arg(ap, int);

        flags = va_arg(ap, int);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "sssdd",

                                    path, name, value, size, flags);

        if (retval > 0) {

            header.size = retval;

            header.type = T_LSETXATTR;

        }

        break;

    case T_LREMOVEXATTR:

        path = va_arg(ap, V9fsString *);

        name = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "ss", path, name);

        if (retval > 0) {

            header.size = retval;

            header.type = T_LREMOVEXATTR;

        }

        break;

    case T_GETVERSION:

        path = va_arg(ap, V9fsString *);

        retval = proxy_marshal(iovec, PROXY_HDR_SZ, "s", path);

        if (retval > 0) {

            header.size = retval;

            header.type = T_GETVERSION;

        }

        break;

    default:

        error_report("Invalid type %d", type);

        retval = -EINVAL;

        break;

    }

    va_end(ap);



    if (retval < 0) {

        goto err_out;

    }



    /* marshal the header details */

    proxy_marshal(iovec, 0, "dd", header.type, header.size);

    header.size += PROXY_HDR_SZ;



    retval = qemu_write_full(proxy->sockfd, iovec->iov_base, header.size);

    if (retval != header.size) {

        goto close_error;

    }



    switch (type) {

    case T_OPEN:

    case T_CREATE:

        /*

         * A file descriptor is returned as response for

         * T_OPEN,T_CREATE on success

         */

        if (v9fs_receivefd(proxy->sockfd, &retval) < 0) {

            goto close_error;

        }

        break;

    case T_MKNOD:

    case T_MKDIR:

    case T_SYMLINK:

    case T_LINK:

    case T_CHMOD:

    case T_CHOWN:

    case T_RENAME:

    case T_TRUNCATE:

    case T_UTIME:

    case T_REMOVE:

    case T_LSETXATTR:

    case T_LREMOVEXATTR:

        if (v9fs_receive_status(proxy, reply, &retval) < 0) {

            goto close_error;

        }

        break;

    case T_LSTAT:

    case T_READLINK:

    case T_STATFS:

    case T_GETVERSION:

        if (v9fs_receive_response(proxy, type, &retval, response) < 0) {

            goto close_error;

        }

        break;

    case T_LGETXATTR:

    case T_LLISTXATTR:

        if (!size) {

            if (v9fs_receive_status(proxy, reply, &retval) < 0) {

                goto close_error;

            }

        } else {

            if (v9fs_receive_response(proxy, type, &retval, response) < 0) {

                goto close_error;

            }

        }

        break;

    }



err_out:

    qemu_mutex_unlock(&proxy->mutex);

    return retval;



close_error:

    close(proxy->sockfd);

    proxy->sockfd = -1;

    qemu_mutex_unlock(&proxy->mutex);

    return -EIO;

}
