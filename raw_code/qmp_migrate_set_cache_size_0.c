void qmp_migrate_set_cache_size(int64_t value, Error **errp)

{

    MigrationState *s = migrate_get_current();



    /* Check for truncation */

    if (value != (size_t)value) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "cache size",

                  "exceeding address space");

        return;

    }



    s->xbzrle_cache_size = xbzrle_cache_resize(value);

}
