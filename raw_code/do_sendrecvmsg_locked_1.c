static abi_long do_sendrecvmsg_locked(int fd, struct target_msghdr *msgp,

                                      int flags, int send)

{

    abi_long ret, len;

    struct msghdr msg;

    int count;

    struct iovec *vec;

    abi_ulong target_vec;



    if (msgp->msg_name) {

        msg.msg_namelen = tswap32(msgp->msg_namelen);

        msg.msg_name = alloca(msg.msg_namelen+1);

        ret = target_to_host_sockaddr(fd, msg.msg_name,

                                      tswapal(msgp->msg_name),

                                      msg.msg_namelen);

        if (ret) {

            goto out2;

        }

    } else {

        msg.msg_name = NULL;

        msg.msg_namelen = 0;

    }

    msg.msg_controllen = 2 * tswapal(msgp->msg_controllen);

    msg.msg_control = alloca(msg.msg_controllen);

    msg.msg_flags = tswap32(msgp->msg_flags);



    count = tswapal(msgp->msg_iovlen);

    target_vec = tswapal(msgp->msg_iov);

    vec = lock_iovec(send ? VERIFY_READ : VERIFY_WRITE,

                     target_vec, count, send);

    if (vec == NULL) {

        ret = -host_to_target_errno(errno);

        goto out2;

    }

    msg.msg_iovlen = count;

    msg.msg_iov = vec;



    if (send) {

        if (fd_trans_target_to_host_data(fd)) {

            ret = fd_trans_target_to_host_data(fd)(msg.msg_iov->iov_base,

                                                   msg.msg_iov->iov_len);

        } else {

            ret = target_to_host_cmsg(&msg, msgp);

        }

        if (ret == 0) {

            ret = get_errno(safe_sendmsg(fd, &msg, flags));

        }

    } else {

        ret = get_errno(safe_recvmsg(fd, &msg, flags));

        if (!is_error(ret)) {

            len = ret;

            if (fd_trans_host_to_target_data(fd)) {

                ret = fd_trans_host_to_target_data(fd)(msg.msg_iov->iov_base,

                                                       len);

            } else {

                ret = host_to_target_cmsg(msgp, &msg);

            }

            if (!is_error(ret)) {

                msgp->msg_namelen = tswap32(msg.msg_namelen);

                if (msg.msg_name != NULL) {

                    ret = host_to_target_sockaddr(tswapal(msgp->msg_name),

                                    msg.msg_name, msg.msg_namelen);

                    if (ret) {

                        goto out;

                    }

                }



                ret = len;

            }

        }

    }



out:

    unlock_iovec(vec, target_vec, count, !send);

out2:

    return ret;

}
