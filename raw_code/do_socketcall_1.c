static abi_long do_socketcall(int num, abi_ulong vptr)

{

    static const unsigned ac[] = { /* number of arguments per call */

        [SOCKOP_socket] = 3,      /* domain, type, protocol */

        [SOCKOP_bind] = 3,        /* sockfd, addr, addrlen */

        [SOCKOP_connect] = 3,     /* sockfd, addr, addrlen */

        [SOCKOP_listen] = 2,      /* sockfd, backlog */

        [SOCKOP_accept] = 3,      /* sockfd, addr, addrlen */

        [SOCKOP_accept4] = 4,     /* sockfd, addr, addrlen, flags */

        [SOCKOP_getsockname] = 3, /* sockfd, addr, addrlen */

        [SOCKOP_getpeername] = 3, /* sockfd, addr, addrlen */

        [SOCKOP_socketpair] = 4,  /* domain, type, protocol, tab */

        [SOCKOP_send] = 4,        /* sockfd, msg, len, flags */

        [SOCKOP_recv] = 4,        /* sockfd, msg, len, flags */

        [SOCKOP_sendto] = 6,      /* sockfd, msg, len, flags, addr, addrlen */

        [SOCKOP_recvfrom] = 6,    /* sockfd, msg, len, flags, addr, addrlen */

        [SOCKOP_shutdown] = 2,    /* sockfd, how */

        [SOCKOP_sendmsg] = 3,     /* sockfd, msg, flags */

        [SOCKOP_recvmsg] = 3,     /* sockfd, msg, flags */

        [SOCKOP_sendmmsg] = 4,    /* sockfd, msgvec, vlen, flags */

        [SOCKOP_recvmmsg] = 4,    /* sockfd, msgvec, vlen, flags */

        [SOCKOP_setsockopt] = 5,  /* sockfd, level, optname, optval, optlen */

        [SOCKOP_getsockopt] = 5,  /* sockfd, level, optname, optval, optlen */

    };

    abi_long a[6]; /* max 6 args */



    /* first, collect the arguments in a[] according to ac[] */

    if (num >= 0 && num < ARRAY_SIZE(ac)) {

        unsigned i;

        assert(ARRAY_SIZE(a) >= ac[num]); /* ensure we have space for args */

        for (i = 0; i < ac[num]; ++i) {

            if (get_user_ual(a[i], vptr + i * sizeof(abi_long)) != 0) {

                return -TARGET_EFAULT;

            }

        }

    }



    /* now when we have the args, actually handle the call */

    switch (num) {

    case SOCKOP_socket: /* domain, type, protocol */

        return do_socket(a[0], a[1], a[2]);

    case SOCKOP_bind: /* sockfd, addr, addrlen */

        return do_bind(a[0], a[1], a[2]);

    case SOCKOP_connect: /* sockfd, addr, addrlen */

        return do_connect(a[0], a[1], a[2]);

    case SOCKOP_listen: /* sockfd, backlog */

        return get_errno(listen(a[0], a[1]));

    case SOCKOP_accept: /* sockfd, addr, addrlen */

        return do_accept4(a[0], a[1], a[2], 0);

    case SOCKOP_accept4: /* sockfd, addr, addrlen, flags */

        return do_accept4(a[0], a[1], a[2], a[3]);

    case SOCKOP_getsockname: /* sockfd, addr, addrlen */

        return do_getsockname(a[0], a[1], a[2]);

    case SOCKOP_getpeername: /* sockfd, addr, addrlen */

        return do_getpeername(a[0], a[1], a[2]);

    case SOCKOP_socketpair: /* domain, type, protocol, tab */

        return do_socketpair(a[0], a[1], a[2], a[3]);

    case SOCKOP_send: /* sockfd, msg, len, flags */

        return do_sendto(a[0], a[1], a[2], a[3], 0, 0);

    case SOCKOP_recv: /* sockfd, msg, len, flags */

        return do_recvfrom(a[0], a[1], a[2], a[3], 0, 0);

    case SOCKOP_sendto: /* sockfd, msg, len, flags, addr, addrlen */

        return do_sendto(a[0], a[1], a[2], a[3], a[4], a[5]);

    case SOCKOP_recvfrom: /* sockfd, msg, len, flags, addr, addrlen */

        return do_recvfrom(a[0], a[1], a[2], a[3], a[4], a[5]);

    case SOCKOP_shutdown: /* sockfd, how */

        return get_errno(shutdown(a[0], a[1]));

    case SOCKOP_sendmsg: /* sockfd, msg, flags */

        return do_sendrecvmsg(a[0], a[1], a[2], 1);

    case SOCKOP_recvmsg: /* sockfd, msg, flags */

        return do_sendrecvmsg(a[0], a[1], a[2], 0);

    case SOCKOP_sendmmsg: /* sockfd, msgvec, vlen, flags */

        return do_sendrecvmmsg(a[0], a[1], a[2], a[3], 1);

    case SOCKOP_recvmmsg: /* sockfd, msgvec, vlen, flags */

        return do_sendrecvmmsg(a[0], a[1], a[2], a[3], 0);

    case SOCKOP_setsockopt: /* sockfd, level, optname, optval, optlen */

        return do_setsockopt(a[0], a[1], a[2], a[3], a[4]);

    case SOCKOP_getsockopt: /* sockfd, level, optname, optval, optlen */

        return do_getsockopt(a[0], a[1], a[2], a[3], a[4]);

    default:

        gemu_log("Unsupported socketcall: %d\n", num);

        return -TARGET_ENOSYS;

    }

}
