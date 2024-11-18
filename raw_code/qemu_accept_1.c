int qemu_accept(int s, struct sockaddr *addr, socklen_t *addrlen)

{

    int ret;



#ifdef CONFIG_ACCEPT4

    ret = accept4(s, addr, addrlen, SOCK_CLOEXEC);

#else

    ret = accept(s, addr, addrlen);

    if (ret >= 0) {

        qemu_set_cloexec(ret);

    }

#endif



    return ret;

}
