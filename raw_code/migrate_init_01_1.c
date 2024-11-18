static MigrationState *migrate_init(const MigrationParams *params)

{

    MigrationState *s = migrate_get_current();

    int64_t bandwidth_limit = s->bandwidth_limit;

    bool enabled_capabilities[MIGRATION_CAPABILITY_MAX];

    int64_t xbzrle_cache_size = s->xbzrle_cache_size;



    memcpy(enabled_capabilities, s->enabled_capabilities,

           sizeof(enabled_capabilities));



    memset(s, 0, sizeof(*s));

    s->params = *params;

    memcpy(s->enabled_capabilities, enabled_capabilities,

           sizeof(enabled_capabilities));

    s->xbzrle_cache_size = xbzrle_cache_size;



    s->bandwidth_limit = bandwidth_limit;

    s->state = MIG_STATE_SETUP;

    trace_migrate_set_state(MIG_STATE_SETUP);



    s->total_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

    return s;

}
