bool migrate_auto_converge(void)

{

    MigrationState *s;



    s = migrate_get_current();



    return s->enabled_capabilities[MIGRATION_CAPABILITY_AUTO_CONVERGE];

}
