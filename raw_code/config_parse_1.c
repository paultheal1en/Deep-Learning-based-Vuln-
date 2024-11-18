static void config_parse(GAConfig *config, int argc, char **argv)

{

    const char *sopt = "hVvdm:p:l:f:F::b:s:t:D";

    int opt_ind = 0, ch;

    const struct option lopt[] = {

        { "help", 0, NULL, 'h' },

        { "version", 0, NULL, 'V' },

        { "dump-conf", 0, NULL, 'D' },

        { "logfile", 1, NULL, 'l' },

        { "pidfile", 1, NULL, 'f' },

#ifdef CONFIG_FSFREEZE

        { "fsfreeze-hook", 2, NULL, 'F' },

#endif

        { "verbose", 0, NULL, 'v' },

        { "method", 1, NULL, 'm' },

        { "path", 1, NULL, 'p' },

        { "daemonize", 0, NULL, 'd' },

        { "blacklist", 1, NULL, 'b' },

#ifdef _WIN32

        { "service", 1, NULL, 's' },

#endif

        { "statedir", 1, NULL, 't' },

        { NULL, 0, NULL, 0 }

    };



    config->log_level = G_LOG_LEVEL_ERROR | G_LOG_LEVEL_CRITICAL;



    while ((ch = getopt_long(argc, argv, sopt, lopt, &opt_ind)) != -1) {

        switch (ch) {

        case 'm':

            g_free(config->method);

            config->method = g_strdup(optarg);

            break;

        case 'p':

            g_free(config->channel_path);

            config->channel_path = g_strdup(optarg);

            break;

        case 'l':

            g_free(config->log_filepath);

            config->log_filepath = g_strdup(optarg);

            break;

        case 'f':

            g_free(config->pid_filepath);

            config->pid_filepath = g_strdup(optarg);

            break;

#ifdef CONFIG_FSFREEZE

        case 'F':

            g_free(config->fsfreeze_hook);

            config->fsfreeze_hook = g_strdup(optarg ?: QGA_FSFREEZE_HOOK_DEFAULT);

            break;

#endif

        case 't':

            g_free(config->state_dir);

            config->state_dir = g_strdup(optarg);

            break;

        case 'v':

            /* enable all log levels */

            config->log_level = G_LOG_LEVEL_MASK;

            break;

        case 'V':

            printf("QEMU Guest Agent %s\n", QEMU_VERSION);

            exit(EXIT_SUCCESS);

        case 'd':

            config->daemonize = 1;

            break;

        case 'D':

            config->dumpconf = 1;

            break;

        case 'b': {

            if (is_help_option(optarg)) {

                qmp_for_each_command(ga_print_cmd, NULL);

                exit(EXIT_SUCCESS);

            }

            config->blacklist = g_list_concat(config->blacklist,

                                             split_list(optarg, ","));

            break;

        }

#ifdef _WIN32

        case 's':

            config->service = optarg;

            if (strcmp(config->service, "install") == 0) {

                if (ga_install_vss_provider()) {

                    exit(EXIT_FAILURE);

                }

                if (ga_install_service(config->channel_path,

                                       config->log_filepath, config->state_dir)) {

                    exit(EXIT_FAILURE);

                }

                exit(EXIT_SUCCESS);

            } else if (strcmp(config->service, "uninstall") == 0) {

                ga_uninstall_vss_provider();

                exit(ga_uninstall_service());

            } else if (strcmp(config->service, "vss-install") == 0) {

                if (ga_install_vss_provider()) {

                    exit(EXIT_FAILURE);

                }

                exit(EXIT_SUCCESS);

            } else if (strcmp(config->service, "vss-uninstall") == 0) {

                ga_uninstall_vss_provider();

                exit(EXIT_SUCCESS);

            } else {

                printf("Unknown service command.\n");

                exit(EXIT_FAILURE);

            }

            break;

#endif

        case 'h':

            usage(argv[0]);

            exit(EXIT_SUCCESS);

        case '?':

            g_print("Unknown option, try '%s --help' for more information.\n",

                    argv[0]);

            exit(EXIT_FAILURE);

        }

    }

}
