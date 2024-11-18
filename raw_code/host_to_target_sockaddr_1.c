static inline abi_long host_to_target_sockaddr(abi_ulong target_addr,

                                               struct sockaddr *addr,

                                               socklen_t len)

{

    struct target_sockaddr *target_saddr;



    if (len == 0) {

        return 0;

    }




    target_saddr = lock_user(VERIFY_WRITE, target_addr, len, 0);

    if (!target_saddr)

        return -TARGET_EFAULT;

    memcpy(target_saddr, addr, len);

    if (len >= offsetof(struct target_sockaddr, sa_family) +

        sizeof(target_saddr->sa_family)) {

        target_saddr->sa_family = tswap16(addr->sa_family);

    }

    if (addr->sa_family == AF_NETLINK && len >= sizeof(struct sockaddr_nl)) {

        struct sockaddr_nl *target_nl = (struct sockaddr_nl *)target_saddr;

        target_nl->nl_pid = tswap32(target_nl->nl_pid);

        target_nl->nl_groups = tswap32(target_nl->nl_groups);

    } else if (addr->sa_family == AF_PACKET) {

        struct sockaddr_ll *target_ll = (struct sockaddr_ll *)target_saddr;

        target_ll->sll_ifindex = tswap32(target_ll->sll_ifindex);

        target_ll->sll_hatype = tswap16(target_ll->sll_hatype);

    } else if (addr->sa_family == AF_INET6 &&

               len >= sizeof(struct target_sockaddr_in6)) {

        struct target_sockaddr_in6 *target_in6 =

               (struct target_sockaddr_in6 *)target_saddr;

        target_in6->sin6_scope_id = tswap16(target_in6->sin6_scope_id);

    }

    unlock_user(target_saddr, target_addr, len);



    return 0;

}