MigrationState *migrate_get_current(void)

{

    static bool once;

    static MigrationState current_migration = {

        .state = MIGRATION_STATUS_NONE,

        .xbzrle_cache_size = DEFAULT_MIGRATE_CACHE_SIZE,

        .mbps = -1,

        .parameters = {

            .compress_level = DEFAULT_MIGRATE_COMPRESS_LEVEL,

            .compress_threads = DEFAULT_MIGRATE_COMPRESS_THREAD_COUNT,

            .decompress_threads = DEFAULT_MIGRATE_DECOMPRESS_THREAD_COUNT,

            .cpu_throttle_initial = DEFAULT_MIGRATE_CPU_THROTTLE_INITIAL,

            .cpu_throttle_increment = DEFAULT_MIGRATE_CPU_THROTTLE_INCREMENT,

            .max_bandwidth = MAX_THROTTLE,

            .downtime_limit = DEFAULT_MIGRATE_SET_DOWNTIME,

            .x_checkpoint_delay = DEFAULT_MIGRATE_X_CHECKPOINT_DELAY,

        },

    };



    if (!once) {

        current_migration.parameters.tls_creds = g_strdup("");

        current_migration.parameters.tls_hostname = g_strdup("");

        once = true;

    }

    return &current_migration;

}
