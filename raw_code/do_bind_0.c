static abi_long do_bind(int sockfd, abi_ulong target_addr,

                        socklen_t addrlen)

{

    void *addr;

    abi_long ret;



    if (addrlen < 0)

        return -TARGET_EINVAL;



    addr = alloca(addrlen+1);



    ret = target_to_host_sockaddr(addr, target_addr, addrlen);

    if (ret)

        return ret;



    return get_errno(bind(sockfd, addr, addrlen));

}
