static inline abi_long target_to_host_sockaddr(int fd, struct sockaddr *addr,

                                               abi_ulong target_addr,

                                               socklen_t len)

{

    const socklen_t unix_maxlen = sizeof (struct sockaddr_un);

    sa_family_t sa_family;

    struct target_sockaddr *target_saddr;



    if (fd_trans_target_to_host_addr(fd)) {

        return fd_trans_target_to_host_addr(fd)(addr, target_addr, len);

    }



    target_saddr = lock_user(VERIFY_READ, target_addr, len, 1);

    if (!target_saddr)

        return -TARGET_EFAULT;



    sa_family = tswap16(target_saddr->sa_family);



    /* Oops. The caller might send a incomplete sun_path; sun_path

     * must be terminated by \0 (see the manual page), but

     * unfortunately it is quite common to specify sockaddr_un

     * length as "strlen(x->sun_path)" while it should be

     * "strlen(...) + 1". We'll fix that here if needed.

     * Linux kernel has a similar feature.

     */



    if (sa_family == AF_UNIX) {

        if (len < unix_maxlen && len > 0) {

            char *cp = (char*)target_saddr;



            if ( cp[len-1] && !cp[len] )

                len++;

        }

        if (len > unix_maxlen)

            len = unix_maxlen;

    }



    memcpy(addr, target_saddr, len);

    addr->sa_family = sa_family;

    if (sa_family == AF_PACKET) {

	struct target_sockaddr_ll *lladdr;



	lladdr = (struct target_sockaddr_ll *)addr;

	lladdr->sll_ifindex = tswap32(lladdr->sll_ifindex);

	lladdr->sll_hatype = tswap16(lladdr->sll_hatype);

    }

    unlock_user(target_saddr, target_addr, 0);



    return 0;

}
