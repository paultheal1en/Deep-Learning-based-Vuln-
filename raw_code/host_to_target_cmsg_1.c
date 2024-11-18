static inline abi_long host_to_target_cmsg(struct target_msghdr *target_msgh,

                                           struct msghdr *msgh)

{

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(msgh);

    abi_long msg_controllen;

    abi_ulong target_cmsg_addr;

    struct target_cmsghdr *target_cmsg, *target_cmsg_start;

    socklen_t space = 0;



    msg_controllen = tswapal(target_msgh->msg_controllen);

    if (msg_controllen < sizeof (struct target_cmsghdr)) 

        goto the_end;

    target_cmsg_addr = tswapal(target_msgh->msg_control);

    target_cmsg = lock_user(VERIFY_WRITE, target_cmsg_addr, msg_controllen, 0);

    target_cmsg_start = target_cmsg;

    if (!target_cmsg)

        return -TARGET_EFAULT;



    while (cmsg && target_cmsg) {

        void *data = CMSG_DATA(cmsg);

        void *target_data = TARGET_CMSG_DATA(target_cmsg);



        int len = cmsg->cmsg_len - CMSG_ALIGN(sizeof (struct cmsghdr));

        int tgt_len, tgt_space;



        /* We never copy a half-header but may copy half-data;

         * this is Linux's behaviour in put_cmsg(). Note that

         * truncation here is a guest problem (which we report

         * to the guest via the CTRUNC bit), unlike truncation

         * in target_to_host_cmsg, which is a QEMU bug.

         */

        if (msg_controllen < sizeof(struct cmsghdr)) {

            target_msgh->msg_flags |= tswap32(MSG_CTRUNC);

            break;

        }



        if (cmsg->cmsg_level == SOL_SOCKET) {

            target_cmsg->cmsg_level = tswap32(TARGET_SOL_SOCKET);

        } else {

            target_cmsg->cmsg_level = tswap32(cmsg->cmsg_level);

        }

        target_cmsg->cmsg_type = tswap32(cmsg->cmsg_type);



        tgt_len = TARGET_CMSG_LEN(len);



        /* Payload types which need a different size of payload on

         * the target must adjust tgt_len here.

         */

        switch (cmsg->cmsg_level) {

        case SOL_SOCKET:

            switch (cmsg->cmsg_type) {

            case SO_TIMESTAMP:

                tgt_len = sizeof(struct target_timeval);

                break;

            default:

                break;

            }

        default:

            break;

        }



        if (msg_controllen < tgt_len) {

            target_msgh->msg_flags |= tswap32(MSG_CTRUNC);

            tgt_len = msg_controllen;

        }



        /* We must now copy-and-convert len bytes of payload

         * into tgt_len bytes of destination space. Bear in mind

         * that in both source and destination we may be dealing

         * with a truncated value!

         */

        switch (cmsg->cmsg_level) {

        case SOL_SOCKET:

            switch (cmsg->cmsg_type) {

            case SCM_RIGHTS:

            {

                int *fd = (int *)data;

                int *target_fd = (int *)target_data;

                int i, numfds = tgt_len / sizeof(int);



                for (i = 0; i < numfds; i++) {

                    __put_user(fd[i], target_fd + i);

                }

                break;

            }

            case SO_TIMESTAMP:

            {

                struct timeval *tv = (struct timeval *)data;

                struct target_timeval *target_tv =

                    (struct target_timeval *)target_data;



                if (len != sizeof(struct timeval) ||

                    tgt_len != sizeof(struct target_timeval)) {

                    goto unimplemented;

                }



                /* copy struct timeval to target */

                __put_user(tv->tv_sec, &target_tv->tv_sec);

                __put_user(tv->tv_usec, &target_tv->tv_usec);

                break;

            }

            case SCM_CREDENTIALS:

            {

                struct ucred *cred = (struct ucred *)data;

                struct target_ucred *target_cred =

                    (struct target_ucred *)target_data;



                __put_user(cred->pid, &target_cred->pid);

                __put_user(cred->uid, &target_cred->uid);

                __put_user(cred->gid, &target_cred->gid);

                break;

            }

            default:

                goto unimplemented;

            }

            break;



        case SOL_IP:

            switch (cmsg->cmsg_type) {

            case IP_TTL:

            {

                uint32_t *v = (uint32_t *)data;

                uint32_t *t_int = (uint32_t *)target_data;



                __put_user(*v, t_int);

                break;

            }

            case IP_RECVERR:

            {

                struct errhdr_t {

                   struct sock_extended_err ee;

                   struct sockaddr_in offender;

                };

                struct errhdr_t *errh = (struct errhdr_t *)data;

                struct errhdr_t *target_errh =

                    (struct errhdr_t *)target_data;



                __put_user(errh->ee.ee_errno, &target_errh->ee.ee_errno);

                __put_user(errh->ee.ee_origin, &target_errh->ee.ee_origin);

                __put_user(errh->ee.ee_type,  &target_errh->ee.ee_type);

                __put_user(errh->ee.ee_code, &target_errh->ee.ee_code);

                __put_user(errh->ee.ee_pad, &target_errh->ee.ee_pad);

                __put_user(errh->ee.ee_info, &target_errh->ee.ee_info);

                __put_user(errh->ee.ee_data, &target_errh->ee.ee_data);

                host_to_target_sockaddr((unsigned long) &target_errh->offender,

                    (void *) &errh->offender, sizeof(errh->offender));

                break;

            }

            default:

                goto unimplemented;

            }

            break;



        case SOL_IPV6:

            switch (cmsg->cmsg_type) {

            case IPV6_HOPLIMIT:

            {

                uint32_t *v = (uint32_t *)data;

                uint32_t *t_int = (uint32_t *)target_data;



                __put_user(*v, t_int);

                break;

            }

            case IPV6_RECVERR:

            {

                struct errhdr6_t {

                   struct sock_extended_err ee;

                   struct sockaddr_in6 offender;

                };

                struct errhdr6_t *errh = (struct errhdr6_t *)data;

                struct errhdr6_t *target_errh =

                    (struct errhdr6_t *)target_data;



                __put_user(errh->ee.ee_errno, &target_errh->ee.ee_errno);

                __put_user(errh->ee.ee_origin, &target_errh->ee.ee_origin);

                __put_user(errh->ee.ee_type,  &target_errh->ee.ee_type);

                __put_user(errh->ee.ee_code, &target_errh->ee.ee_code);

                __put_user(errh->ee.ee_pad, &target_errh->ee.ee_pad);

                __put_user(errh->ee.ee_info, &target_errh->ee.ee_info);

                __put_user(errh->ee.ee_data, &target_errh->ee.ee_data);

                host_to_target_sockaddr((unsigned long) &target_errh->offender,

                    (void *) &errh->offender, sizeof(errh->offender));

                break;

            }

            default:

                goto unimplemented;

            }

            break;



        default:

        unimplemented:

            gemu_log("Unsupported ancillary data: %d/%d\n",

                                        cmsg->cmsg_level, cmsg->cmsg_type);

            memcpy(target_data, data, MIN(len, tgt_len));

            if (tgt_len > len) {

                memset(target_data + len, 0, tgt_len - len);

            }

        }



        target_cmsg->cmsg_len = tswapal(tgt_len);

        tgt_space = TARGET_CMSG_SPACE(len);

        if (msg_controllen < tgt_space) {

            tgt_space = msg_controllen;

        }

        msg_controllen -= tgt_space;

        space += tgt_space;

        cmsg = CMSG_NXTHDR(msgh, cmsg);

        target_cmsg = TARGET_CMSG_NXTHDR(target_msgh, target_cmsg,

                                         target_cmsg_start);

    }

    unlock_user(target_cmsg, target_cmsg_addr, space);

 the_end:

    target_msgh->msg_controllen = tswapal(space);

    return 0;

}
