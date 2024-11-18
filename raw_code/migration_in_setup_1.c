bool migration_in_setup(MigrationState *s)

{

    return s->state == MIG_STATE_SETUP;

}
