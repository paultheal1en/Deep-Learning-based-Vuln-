void qemu_spice_init(void)

{

    QemuOpts *opts = QTAILQ_FIRST(&qemu_spice_opts.head);

    const char *password, *str, *x509_dir, *addr,

        *x509_key_password = NULL,

        *x509_dh_file = NULL,

        *tls_ciphers = NULL;

    char *x509_key_file = NULL,

        *x509_cert_file = NULL,

        *x509_cacert_file = NULL;

    int port, tls_port, len, addr_flags;

    spice_image_compression_t compression;

    spice_wan_compression_t wan_compr;



    qemu_thread_get_self(&me);



   if (!opts) {

        return;

    }

    port = qemu_opt_get_number(opts, "port", 0);

    tls_port = qemu_opt_get_number(opts, "tls-port", 0);

    if (!port && !tls_port) {

        fprintf(stderr, "neither port nor tls-port specified for spice.");

        exit(1);

    }

    if (port < 0 || port > 65535) {

        fprintf(stderr, "spice port is out of range");

        exit(1);

    }

    if (tls_port < 0 || tls_port > 65535) {

        fprintf(stderr, "spice tls-port is out of range");

        exit(1);

    }

    password = qemu_opt_get(opts, "password");



    if (tls_port) {

        x509_dir = qemu_opt_get(opts, "x509-dir");

        if (NULL == x509_dir) {

            x509_dir = ".";

        }

        len = strlen(x509_dir) + 32;



        str = qemu_opt_get(opts, "x509-key-file");

        if (str) {

            x509_key_file = g_strdup(str);

        } else {

            x509_key_file = g_malloc(len);

            snprintf(x509_key_file, len, "%s/%s", x509_dir, X509_SERVER_KEY_FILE);

        }



        str = qemu_opt_get(opts, "x509-cert-file");

        if (str) {

            x509_cert_file = g_strdup(str);

        } else {

            x509_cert_file = g_malloc(len);

            snprintf(x509_cert_file, len, "%s/%s", x509_dir, X509_SERVER_CERT_FILE);

        }



        str = qemu_opt_get(opts, "x509-cacert-file");

        if (str) {

            x509_cacert_file = g_strdup(str);

        } else {

            x509_cacert_file = g_malloc(len);

            snprintf(x509_cacert_file, len, "%s/%s", x509_dir, X509_CA_CERT_FILE);

        }



        x509_key_password = qemu_opt_get(opts, "x509-key-password");

        x509_dh_file = qemu_opt_get(opts, "x509-dh-file");

        tls_ciphers = qemu_opt_get(opts, "tls-ciphers");

    }



    addr = qemu_opt_get(opts, "addr");

    addr_flags = 0;

    if (qemu_opt_get_bool(opts, "ipv4", 0)) {

        addr_flags |= SPICE_ADDR_FLAG_IPV4_ONLY;

    } else if (qemu_opt_get_bool(opts, "ipv6", 0)) {

        addr_flags |= SPICE_ADDR_FLAG_IPV6_ONLY;

    }



    spice_server = spice_server_new();

    spice_server_set_addr(spice_server, addr ? addr : "", addr_flags);

    if (port) {

        spice_server_set_port(spice_server, port);

    }

    if (tls_port) {

        spice_server_set_tls(spice_server, tls_port,

                             x509_cacert_file,

                             x509_cert_file,

                             x509_key_file,

                             x509_key_password,

                             x509_dh_file,

                             tls_ciphers);

    }

    if (password) {

        spice_server_set_ticket(spice_server, password, 0, 0, 0);

    }

    if (qemu_opt_get_bool(opts, "sasl", 0)) {

#if SPICE_SERVER_VERSION >= 0x000900 /* 0.9.0 */

        if (spice_server_set_sasl_appname(spice_server, "qemu") == -1 ||

            spice_server_set_sasl(spice_server, 1) == -1) {

            fprintf(stderr, "spice: failed to enable sasl\n");

            exit(1);

        }

#else

        fprintf(stderr, "spice: sasl is not available (spice >= 0.9 required)\n");

        exit(1);

#endif

    }

    if (qemu_opt_get_bool(opts, "disable-ticketing", 0)) {

        auth = "none";

        spice_server_set_noauth(spice_server);

    }



#if SPICE_SERVER_VERSION >= 0x000801

    if (qemu_opt_get_bool(opts, "disable-copy-paste", 0)) {

        spice_server_set_agent_copypaste(spice_server, false);

    }

#endif



    compression = SPICE_IMAGE_COMPRESS_AUTO_GLZ;

    str = qemu_opt_get(opts, "image-compression");

    if (str) {

        compression = parse_compression(str);

    }

    spice_server_set_image_compression(spice_server, compression);



    wan_compr = SPICE_WAN_COMPRESSION_AUTO;

    str = qemu_opt_get(opts, "jpeg-wan-compression");

    if (str) {

        wan_compr = parse_wan_compression(str);

    }

    spice_server_set_jpeg_compression(spice_server, wan_compr);



    wan_compr = SPICE_WAN_COMPRESSION_AUTO;

    str = qemu_opt_get(opts, "zlib-glz-wan-compression");

    if (str) {

        wan_compr = parse_wan_compression(str);

    }

    spice_server_set_zlib_glz_compression(spice_server, wan_compr);



    str = qemu_opt_get(opts, "streaming-video");

    if (str) {

        int streaming_video = parse_stream_video(str);

        spice_server_set_streaming_video(spice_server, streaming_video);

    }



    spice_server_set_agent_mouse

        (spice_server, qemu_opt_get_bool(opts, "agent-mouse", 1));

    spice_server_set_playback_compression

        (spice_server, qemu_opt_get_bool(opts, "playback-compression", 1));



    qemu_opt_foreach(opts, add_channel, NULL, 0);



    if (0 != spice_server_init(spice_server, &core_interface)) {

        fprintf(stderr, "failed to initialize spice server");

        exit(1);

    };

    using_spice = 1;



    migration_state.notify = migration_state_notifier;

    add_migration_state_change_notifier(&migration_state);

#ifdef SPICE_INTERFACE_MIGRATION

    spice_migrate.sin.base.sif = &migrate_interface.base;

    spice_migrate.connect_complete.cb = NULL;

    qemu_spice_add_interface(&spice_migrate.sin.base);

#endif



    qemu_spice_input_init();

    qemu_spice_audio_init();



    g_free(x509_key_file);

    g_free(x509_cert_file);

    g_free(x509_cacert_file);

}
