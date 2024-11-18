static SocketAddress *tcp_build_address(const char *host_port, Error **errp)

{

    InetSocketAddress *iaddr = g_new(InetSocketAddress, 1);

    SocketAddress *saddr;



    if (inet_parse(iaddr, host_port, errp)) {

        qapi_free_InetSocketAddress(iaddr);

        return NULL;

    }



    saddr = g_new0(SocketAddress, 1);

    saddr->type = SOCKET_ADDRESS_KIND_INET;

    saddr->u.inet.data = iaddr;



    return saddr;

}
