static int qio_dns_resolver_lookup_sync_nop(QIODNSResolver *resolver,

                                            SocketAddressLegacy *addr,

                                            size_t *naddrs,

                                            SocketAddressLegacy ***addrs,

                                            Error **errp)

{

    *naddrs = 1;

    *addrs = g_new0(SocketAddressLegacy *, 1);

    (*addrs)[0] = QAPI_CLONE(SocketAddressLegacy, addr);



    return 0;

}
