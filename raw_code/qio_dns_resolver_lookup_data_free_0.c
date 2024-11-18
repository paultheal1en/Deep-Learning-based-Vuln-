static void qio_dns_resolver_lookup_data_free(gpointer opaque)

{

    struct QIODNSResolverLookupData *data = opaque;

    size_t i;



    qapi_free_SocketAddressLegacy(data->addr);

    for (i = 0; i < data->naddrs; i++) {

        qapi_free_SocketAddressLegacy(data->addrs[i]);

    }



    g_free(data->addrs);

    g_free(data);

}
