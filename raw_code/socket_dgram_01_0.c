int socket_dgram(SocketAddressLegacy *remote, SocketAddressLegacy *local, Error **errp)

{

    int fd;



    /*

     * TODO SOCKET_ADDRESS_LEGACY_KIND_FD when fd is AF_INET or AF_INET6

     * (although other address families can do SOCK_DGRAM, too)

     */

    switch (remote->type) {

    case SOCKET_ADDRESS_LEGACY_KIND_INET:

        fd = inet_dgram_saddr(remote->u.inet.data,

                              local ? local->u.inet.data : NULL, errp);

        break;



    default:

        error_setg(errp, "socket type unsupported for datagram");

        fd = -1;

    }

    return fd;

}
