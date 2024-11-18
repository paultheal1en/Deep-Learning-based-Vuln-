static void migrate_params_test_apply(MigrateSetParameters *params,

                                      MigrationParameters *dest)

{

    *dest = migrate_get_current()->parameters;



    /* TODO use QAPI_CLONE() instead of duplicating it inline */



    if (params->has_compress_level) {

        dest->compress_level = params->compress_level;

    }



    if (params->has_compress_threads) {

        dest->compress_threads = params->compress_threads;

    }



    if (params->has_decompress_threads) {

        dest->decompress_threads = params->decompress_threads;

    }



    if (params->has_cpu_throttle_initial) {

        dest->cpu_throttle_initial = params->cpu_throttle_initial;

    }



    if (params->has_cpu_throttle_increment) {

        dest->cpu_throttle_increment = params->cpu_throttle_increment;

    }



    if (params->has_tls_creds) {

        dest->tls_creds = g_strdup(params->tls_creds);

    }



    if (params->has_tls_hostname) {

        dest->tls_hostname = g_strdup(params->tls_hostname);

    }



    if (params->has_max_bandwidth) {

        dest->max_bandwidth = params->max_bandwidth;

    }



    if (params->has_downtime_limit) {

        dest->downtime_limit = params->downtime_limit;

    }



    if (params->has_x_checkpoint_delay) {

        dest->x_checkpoint_delay = params->x_checkpoint_delay;

    }



    if (params->has_block_incremental) {

        dest->block_incremental = params->block_incremental;

    }

}
