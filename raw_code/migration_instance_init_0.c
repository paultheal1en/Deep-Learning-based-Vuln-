static void migration_instance_init(Object *obj)

{

    MigrationState *ms = MIGRATION_OBJ(obj);



    ms->state = MIGRATION_STATUS_NONE;

    ms->xbzrle_cache_size = DEFAULT_MIGRATE_CACHE_SIZE;

    ms->mbps = -1;

    ms->parameters.tls_creds = g_strdup("");

    ms->parameters.tls_hostname = g_strdup("");

}
