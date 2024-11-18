bool migrate_rdma_pin_all(void)

{

    MigrationState *s;



    s = migrate_get_current();



    return s->enabled_capabilities[MIGRATION_CAPABILITY_RDMA_PIN_ALL];

}
