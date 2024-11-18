static void smp_parse(QemuOpts *opts)

{

    if (opts) {



        unsigned cpus    = qemu_opt_get_number(opts, "cpus", 0);

        unsigned sockets = qemu_opt_get_number(opts, "sockets", 0);

        unsigned cores   = qemu_opt_get_number(opts, "cores", 0);

        unsigned threads = qemu_opt_get_number(opts, "threads", 0);



        /* compute missing values, prefer sockets over cores over threads */

        if (cpus == 0 || sockets == 0) {

            sockets = sockets > 0 ? sockets : 1;

            cores = cores > 0 ? cores : 1;

            threads = threads > 0 ? threads : 1;

            if (cpus == 0) {

                cpus = cores * threads * sockets;

            }

        } else if (cores == 0) {

            threads = threads > 0 ? threads : 1;

            cores = cpus / (sockets * threads);

        } else if (threads == 0) {

            threads = cpus / (cores * sockets);

        } else if (sockets * cores * threads < cpus) {

            error_report("cpu topology: "

                         "sockets (%u) * cores (%u) * threads (%u) < "

                         "smp_cpus (%u)",

                         sockets, cores, threads, cpus);

            exit(1);

        }



        max_cpus = qemu_opt_get_number(opts, "maxcpus", cpus);

        if (sockets * cores * threads > max_cpus) {

            error_report("cpu topology: "

                         "sockets (%u) * cores (%u) * threads (%u) > "

                         "maxcpus (%u)",

                         sockets, cores, threads, max_cpus);

            exit(1);

        }



        smp_cpus = cpus;

        smp_cores = cores > 0 ? cores : 1;

        smp_threads = threads > 0 ? threads : 1;



    }



    if (max_cpus == 0) {

        max_cpus = smp_cpus;

    }



    if (max_cpus > MAX_CPUMASK_BITS) {

        error_report("unsupported number of maxcpus");

        exit(1);

    }

    if (max_cpus < smp_cpus) {

        error_report("maxcpus must be equal to or greater than smp");

        exit(1);

    }



    if (smp_cpus > 1 || smp_cores > 1 || smp_threads > 1) {

        Error *blocker = NULL;

        error_setg(&blocker, QERR_REPLAY_NOT_SUPPORTED, "smp");

        replay_add_blocker(blocker);

    }

}
