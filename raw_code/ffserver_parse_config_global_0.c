static int ffserver_parse_config_global(FFServerConfig *config, const char *cmd,

                                        const char **p, int line_num)

{

    int val;

    char arg[1024];

    if (!av_strcasecmp(cmd, "Port") || !av_strcasecmp(cmd, "HTTPPort")) {

        if (!av_strcasecmp(cmd, "Port"))

            WARNING("Port option is deprecated, use HTTPPort instead\n");

        ffserver_get_arg(arg, sizeof(arg), p);

        val = atoi(arg);

        if (val < 1 || val > 65536)

            ERROR("Invalid port: %s\n", arg);

        if (val < 1024)

            WARNING("Trying to use IETF assigned system port: %d\n", val);

        config->http_addr.sin_port = htons(val);

    } else if (!av_strcasecmp(cmd, "HTTPBindAddress") || !av_strcasecmp(cmd, "BindAddress")) {

        if (!av_strcasecmp(cmd, "BindAddress"))

            WARNING("BindAddress option is deprecated, use HTTPBindAddress instead\n");

        ffserver_get_arg(arg, sizeof(arg), p);

        if (resolve_host(&config->http_addr.sin_addr, arg) != 0)

            ERROR("%s:%d: Invalid host/IP address: %s\n", arg);

    } else if (!av_strcasecmp(cmd, "NoDaemon")) {

        WARNING("NoDaemon option has no effect, you should remove it\n");

    } else if (!av_strcasecmp(cmd, "RTSPPort")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        val = atoi(arg);

        if (val < 1 || val > 65536)

            ERROR("%s:%d: Invalid port: %s\n", arg);

        config->rtsp_addr.sin_port = htons(atoi(arg));

    } else if (!av_strcasecmp(cmd, "RTSPBindAddress")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        if (resolve_host(&config->rtsp_addr.sin_addr, arg) != 0)

            ERROR("Invalid host/IP address: %s\n", arg);

    } else if (!av_strcasecmp(cmd, "MaxHTTPConnections")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        val = atoi(arg);

        if (val < 1 || val > 65536)

            ERROR("Invalid MaxHTTPConnections: %s\n", arg);

        config->nb_max_http_connections = val;

    } else if (!av_strcasecmp(cmd, "MaxClients")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        val = atoi(arg);

        if (val < 1 || val > config->nb_max_http_connections)

            ERROR("Invalid MaxClients: %s\n", arg);

        else

            config->nb_max_connections = val;

    } else if (!av_strcasecmp(cmd, "MaxBandwidth")) {

        int64_t llval;

        ffserver_get_arg(arg, sizeof(arg), p);

        llval = strtoll(arg, NULL, 10);

        if (llval < 10 || llval > 10000000)

            ERROR("Invalid MaxBandwidth: %s\n", arg);

        else

            config->max_bandwidth = llval;

    } else if (!av_strcasecmp(cmd, "CustomLog")) {

        if (!config->debug)

            ffserver_get_arg(config->logfilename, sizeof(config->logfilename), p);

    } else if (!av_strcasecmp(cmd, "LoadModule")) {

        ERROR("Loadable modules no longer supported\n");

    } else

        ERROR("Incorrect keyword: '%s'\n", cmd);

    return 0;

}
