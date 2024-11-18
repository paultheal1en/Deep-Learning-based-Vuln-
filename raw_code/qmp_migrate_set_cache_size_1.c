void qmp_migrate_set_cache_size(int64_t value, Error **errp)

{

    MigrationState *s = migrate_get_current();

    int64_t new_size;



    /* Check for truncation */

    if (value != (size_t)value) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "cache size",

                  "exceeding address space");

        return;

    }



    /* Cache should not be larger than guest ram size */

    if (value > ram_bytes_total()) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "cache size",

                  "exceeds guest ram size ");

        return;

    }



    new_size = xbzrle_cache_resize(value);

    if (new_size < 0) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "cache size",

                  "is smaller than page size");

        return;

    }



    s->xbzrle_cache_size = new_size;

}
