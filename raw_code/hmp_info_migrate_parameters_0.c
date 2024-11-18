void hmp_info_migrate_parameters(Monitor *mon, const QDict *qdict)

{

    MigrationParameters *params;



    params = qmp_query_migrate_parameters(NULL);



    if (params) {

        assert(params->has_compress_level);

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_COMPRESS_LEVEL),

            params->compress_level);

        assert(params->has_compress_threads);

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_COMPRESS_THREADS),

            params->compress_threads);

        assert(params->has_decompress_threads);

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_DECOMPRESS_THREADS),

            params->decompress_threads);

        assert(params->has_cpu_throttle_initial);

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_CPU_THROTTLE_INITIAL),

            params->cpu_throttle_initial);

        assert(params->has_cpu_throttle_increment);

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_CPU_THROTTLE_INCREMENT),

            params->cpu_throttle_increment);

        assert(params->has_tls_creds);

        monitor_printf(mon, "%s: '%s'\n",

            MigrationParameter_str(MIGRATION_PARAMETER_TLS_CREDS),

            params->tls_creds);

        assert(params->has_tls_hostname);

        monitor_printf(mon, "%s: '%s'\n",

            MigrationParameter_str(MIGRATION_PARAMETER_TLS_HOSTNAME),

            params->tls_hostname);

        assert(params->has_max_bandwidth);

        monitor_printf(mon, "%s: %" PRId64 " bytes/second\n",

            MigrationParameter_str(MIGRATION_PARAMETER_MAX_BANDWIDTH),

            params->max_bandwidth);

        assert(params->has_downtime_limit);

        monitor_printf(mon, "%s: %" PRId64 " milliseconds\n",

            MigrationParameter_str(MIGRATION_PARAMETER_DOWNTIME_LIMIT),

            params->downtime_limit);

        assert(params->has_x_checkpoint_delay);

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_X_CHECKPOINT_DELAY),

            params->x_checkpoint_delay);

        assert(params->has_block_incremental);

        monitor_printf(mon, "%s: %s\n",

            MigrationParameter_str(MIGRATION_PARAMETER_BLOCK_INCREMENTAL),

            params->block_incremental ? "on" : "off");

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_X_MULTIFD_CHANNELS),

            params->x_multifd_channels);

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_X_MULTIFD_PAGE_COUNT),

            params->x_multifd_page_count);

        monitor_printf(mon, "%s: %" PRId64 "\n",

            MigrationParameter_str(MIGRATION_PARAMETER_XBZRLE_CACHE_SIZE),

            params->xbzrle_cache_size);

    }



    qapi_free_MigrationParameters(params);

}
