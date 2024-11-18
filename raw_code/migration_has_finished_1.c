bool migration_has_finished(MigrationState *s)

{

    return s->state == MIG_STATE_COMPLETED;

}
