ivshmem_server_parse_args(IvshmemServerArgs *args, int argc, char *argv[])

{

    int c;

    unsigned long long v;

    Error *errp = NULL;



    while ((c = getopt(argc, argv,

                       "h"  /* help */

                       "v"  /* verbose */

                       "F"  /* foreground */

                       "p:" /* pid_file */

                       "S:" /* unix_socket_path */

                       "m:" /* shm_path */

                       "l:" /* shm_size */

                       "n:" /* n_vectors */

                      )) != -1) {



        switch (c) {

        case 'h': /* help */

            ivshmem_server_usage(argv[0], 0);

            break;



        case 'v': /* verbose */

            args->verbose = 1;

            break;



        case 'F': /* foreground */

            args->foreground = 1;

            break;



        case 'p': /* pid_file */

            args->pid_file = strdup(optarg);

            break;



        case 'S': /* unix_socket_path */

            args->unix_socket_path = strdup(optarg);

            break;



        case 'm': /* shm_path */

            args->shm_path = strdup(optarg);

            break;



        case 'l': /* shm_size */

            parse_option_size("shm_size", optarg, &args->shm_size, &errp);

            if (errp) {

                fprintf(stderr, "cannot parse shm size: %s\n",

                        error_get_pretty(errp));

                error_free(errp);

                ivshmem_server_usage(argv[0], 1);

            }

            break;



        case 'n': /* n_vectors */

            if (parse_uint_full(optarg, &v, 0) < 0) {

                fprintf(stderr, "cannot parse n_vectors\n");

                ivshmem_server_usage(argv[0], 1);

            }

            args->n_vectors = v;

            break;



        default:

            ivshmem_server_usage(argv[0], 1);

            break;

        }

    }



    if (args->n_vectors > IVSHMEM_SERVER_MAX_VECTORS) {

        fprintf(stderr, "too many requested vectors (max is %d)\n",

                IVSHMEM_SERVER_MAX_VECTORS);

        ivshmem_server_usage(argv[0], 1);

    }



    if (args->verbose == 1 && args->foreground == 0) {

        fprintf(stderr, "cannot use verbose in daemon mode\n");

        ivshmem_server_usage(argv[0], 1);

    }

}
