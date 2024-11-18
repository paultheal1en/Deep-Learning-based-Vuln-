void qmp_migrate_set_parameters(bool has_compress_level,

                                int64_t compress_level,

                                bool has_compress_threads,

                                int64_t compress_threads,

                                bool has_decompress_threads,

                                int64_t decompress_threads,

                                bool has_cpu_throttle_initial,

                                int64_t cpu_throttle_initial,

                                bool has_cpu_throttle_increment,

                                int64_t cpu_throttle_increment,





                                Error **errp)

{

    MigrationState *s = migrate_get_current();



    if (has_compress_level && (compress_level < 0 || compress_level > 9)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "compress_level",

                   "is invalid, it should be in the range of 0 to 9");

        return;

    }

    if (has_compress_threads &&

            (compress_threads < 1 || compress_threads > 255)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "compress_threads",

                   "is invalid, it should be in the range of 1 to 255");

        return;

    }

    if (has_decompress_threads &&

            (decompress_threads < 1 || decompress_threads > 255)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "decompress_threads",

                   "is invalid, it should be in the range of 1 to 255");

        return;

    }

    if (has_cpu_throttle_initial &&

            (cpu_throttle_initial < 1 || cpu_throttle_initial > 99)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "cpu_throttle_initial",

                   "an integer in the range of 1 to 99");

    }

    if (has_cpu_throttle_increment &&

            (cpu_throttle_increment < 1 || cpu_throttle_increment > 99)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                   "cpu_throttle_increment",

                   "an integer in the range of 1 to 99");

    }



    if (has_compress_level) {

        s->parameters.compress_level = compress_level;

    }

    if (has_compress_threads) {

        s->parameters.compress_threads = compress_threads;

    }

    if (has_decompress_threads) {

        s->parameters.decompress_threads = decompress_threads;

    }

    if (has_cpu_throttle_initial) {

        s->parameters.cpu_throttle_initial = cpu_throttle_initial;

    }

    if (has_cpu_throttle_increment) {

        s->parameters.cpu_throttle_increment = cpu_throttle_increment;

    }

    if (has_tls_creds) {

        g_free(s->parameters.tls_creds);

        s->parameters.tls_creds = g_strdup(tls_creds);

    }

    if (has_tls_hostname) {

        g_free(s->parameters.tls_hostname);

        s->parameters.tls_hostname = g_strdup(tls_hostname);

    }

}