static void migrate_params_apply(MigrateSetParameters *params)

{

    MigrationState *s = migrate_get_current();



    /* TODO use QAPI_CLONE() instead of duplicating it inline */



    if (params->has_compress_level) {

        s->parameters.compress_level = params->compress_level;

    }



    if (params->has_compress_threads) {

        s->parameters.compress_threads = params->compress_threads;

    }



    if (params->has_decompress_threads) {

        s->parameters.decompress_threads = params->decompress_threads;

    }



    if (params->has_cpu_throttle_initial) {

        s->parameters.cpu_throttle_initial = params->cpu_throttle_initial;

    }



    if (params->has_cpu_throttle_increment) {

        s->parameters.cpu_throttle_increment = params->cpu_throttle_increment;

    }



    if (params->has_tls_creds) {

        g_free(s->parameters.tls_creds);

        s->parameters.tls_creds = g_strdup(params->tls_creds);

    }



    if (params->has_tls_hostname) {

        g_free(s->parameters.tls_hostname);

        s->parameters.tls_hostname = g_strdup(params->tls_hostname);

    }



    if (params->has_max_bandwidth) {

        s->parameters.max_bandwidth = params->max_bandwidth;

        if (s->to_dst_file) {

            qemu_file_set_rate_limit(s->to_dst_file,

                                s->parameters.max_bandwidth / XFER_LIMIT_RATIO);

        }

    }



    if (params->has_downtime_limit) {

        s->parameters.downtime_limit = params->downtime_limit;

    }



    if (params->has_x_checkpoint_delay) {

        s->parameters.x_checkpoint_delay = params->x_checkpoint_delay;

        if (migration_in_colo_state()) {

            colo_checkpoint_notify(s);

        }

    }



    if (params->has_block_incremental) {

        s->parameters.block_incremental = params->block_incremental;

    }

}
