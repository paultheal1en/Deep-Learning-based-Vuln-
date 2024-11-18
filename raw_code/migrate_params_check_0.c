static bool migrate_params_check(MigrationParameters *params, Error **errp)

{

    if (params->has_compress_level &&

        (params->compress_level < 0 || params->compress_level > 9)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "compress_level",

                   "is invalid, it should be in the range of 0 to 9");

        return false;

    }



    if (params->has_compress_threads &&

        (params->compress_threads < 1 || params->compress_threads > 255)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "compress_threads",

                   "is invalid, it should be in the range of 1 to 255");

        return false;

    }



    if (params->has_decompress_threads &&

        (params->decompress_threads < 1 || params->decompress_threads > 255)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "decompress_threads",

                   "is invalid, it should be in the range of 1 to 255");

        return false;

    }



    if (params->has_cpu_throttle_initial &&

        (params->cpu_throttle_initial < 1 ||

         params->cpu_throttle_initial > 99)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "cpu_throttle_initial",

                   "an integer in the range of 1 to 99");

        return false;

    }



    if (params->has_cpu_throttle_increment &&

        (params->cpu_throttle_increment < 1 ||

         params->cpu_throttle_increment > 99)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "cpu_throttle_increment",

                   "an integer in the range of 1 to 99");

        return false;

    }



    if (params->has_max_bandwidth &&

        (params->max_bandwidth < 0 || params->max_bandwidth > SIZE_MAX)) {

        error_setg(errp, "Parameter 'max_bandwidth' expects an integer in the"

                         " range of 0 to %zu bytes/second", SIZE_MAX);

        return false;

    }



    if (params->has_downtime_limit &&

        (params->downtime_limit < 0 ||

         params->downtime_limit > MAX_MIGRATE_DOWNTIME)) {

        error_setg(errp, "Parameter 'downtime_limit' expects an integer in "

                         "the range of 0 to %d milliseconds",

                         MAX_MIGRATE_DOWNTIME);

        return false;

    }



    if (params->has_x_checkpoint_delay && (params->x_checkpoint_delay < 0)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                    "x_checkpoint_delay",

                    "is invalid, it should be positive");

        return false;

    }

    if (params->has_x_multifd_channels &&

        (params->x_multifd_channels < 1 || params->x_multifd_channels > 255)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "multifd_channels",

                   "is invalid, it should be in the range of 1 to 255");

        return false;

    }

    if (params->has_x_multifd_page_count &&

            (params->x_multifd_page_count < 1 ||

             params->x_multifd_page_count > 10000)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "multifd_page_count",

                   "is invalid, it should be in the range of 1 to 10000");

        return false;

    }



    if (params->has_xbzrle_cache_size &&

        (params->xbzrle_cache_size < qemu_target_page_size() ||

         !is_power_of_2(params->xbzrle_cache_size))) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "xbzrle_cache_size",

                   "is invalid, it should be bigger than target page size"

                   " and a power of two");

        return false;

    }



    return true;

}
