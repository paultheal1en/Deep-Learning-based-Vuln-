int avformat_network_init(void)

{

#if CONFIG_NETWORK

    int ret;

    ff_network_inited_globally = 1;

    if ((ret = ff_network_init()) < 0)

        return ret;

    ff_tls_init();

#endif

    return 0;

}
