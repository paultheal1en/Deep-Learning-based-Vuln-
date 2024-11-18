static abi_long do_sendto(int fd, abi_ulong msg, size_t len, int flags,

                          abi_ulong target_addr, socklen_t addrlen)

{

    void *addr;

    void *host_msg;

    abi_long ret;



    if (addrlen < 0 || addrlen > MAX_SOCK_ADDR)

        return -TARGET_EINVAL;



    host_msg = lock_user(VERIFY_READ, msg, len, 1);

    if (!host_msg)

        return -TARGET_EFAULT;

    if (target_addr) {

        addr = alloca(addrlen);

        target_to_host_sockaddr(addr, target_addr, addrlen);

        ret = get_errno(sendto(fd, host_msg, len, flags, addr, addrlen));

    } else {

        ret = get_errno(send(fd, host_msg, len, flags));

    }

    unlock_user(host_msg, msg, 0);

    return ret;

}
