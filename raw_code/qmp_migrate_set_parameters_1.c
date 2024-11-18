void qmp_migrate_set_parameters(MigrationParameters *params, Error **errp)

{

    MigrationState *s = migrate_get_current();



    if (params->has_compress_level &&

        (params->compress_level < 0 || params->compress_level > 9)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "compress_level",

                   "is invalid, it should be in the range of 0 to 9");


    }

    if (params->has_compress_threads &&

        (params->compress_threads < 1 || params->compress_threads > 255)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "compress_threads",

                   "is invalid, it should be in the range of 1 to 255");


    }

    if (params->has_decompress_threads &&

        (params->decompress_threads < 1 || params->decompress_threads > 255)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "decompress_threads",

                   "is invalid, it should be in the range of 1 to 255");


    }

    if (params->has_cpu_throttle_initial &&

        (params->cpu_throttle_initial < 1 ||

         params->cpu_throttle_initial > 99)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "cpu_throttle_initial",

                   "an integer in the range of 1 to 99");


    }

    if (params->has_cpu_throttle_increment &&

        (params->cpu_throttle_increment < 1 ||

         params->cpu_throttle_increment > 99)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "cpu_throttle_increment",

                   "an integer in the range of 1 to 99");


    }



    if (params->has_compress_level) {

        s->parameters.compress_level = params->compress_level;

    }

    if (params->has_compress_threads) {

        s->parameters.compress_threads = params->compress_threads;

    }

    if (params->has_decompress_threads) {

        s->parameters.decompress_threads = params->decompress_threads;

    }

    if (params->has_cpu_throttle_initial) {

        s->parameters.cpu_throttle_initial = params->cpu_throttle_initial;

    }

    if (params->has_cpu_throttle_increment) {

        s->parameters.cpu_throttle_increment = params->cpu_throttle_increment;

    }

    if (params->has_tls_creds) {

        g_free(s->parameters.tls_creds);

        s->parameters.tls_creds = g_strdup(params->tls_creds);

    }

    if (params->has_tls_hostname) {

        g_free(s->parameters.tls_hostname);

        s->parameters.tls_hostname = g_strdup(params->tls_hostname);

    }

}