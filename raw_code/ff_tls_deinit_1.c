void ff_tls_deinit(void)

{

#if CONFIG_TLS_OPENSSL_PROTOCOL

    ff_openssl_deinit();

#endif

#if CONFIG_TLS_GNUTLS_PROTOCOL

    ff_gnutls_deinit();

#endif

}
