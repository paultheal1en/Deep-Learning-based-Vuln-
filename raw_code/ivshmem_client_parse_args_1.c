ivshmem_client_parse_args(IvshmemClientArgs *args, int argc, char *argv[])

{

    int c;



    while ((c = getopt(argc, argv,

                       "h"  /* help */

                       "v"  /* verbose */

                       "S:" /* unix_sock_path */

                      )) != -1) {



        switch (c) {

        case 'h': /* help */

            ivshmem_client_usage(argv[0], 0);

            break;



        case 'v': /* verbose */

            args->verbose = 1;

            break;



        case 'S': /* unix_sock_path */

            args->unix_sock_path = strdup(optarg);

            break;



        default:

            ivshmem_client_usage(argv[0], 1);

            break;

        }

    }

}
