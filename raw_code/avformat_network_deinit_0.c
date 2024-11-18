int avformat_network_deinit(void)

{

#if CONFIG_NETWORK

    ff_network_close();

    ff_tls_deinit();

    ff_network_inited_globally = 0;

#endif

    return 0;

}
