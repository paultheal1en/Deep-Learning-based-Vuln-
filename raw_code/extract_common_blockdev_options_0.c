static void extract_common_blockdev_options(QemuOpts *opts, int *bdrv_flags,

    const char **throttling_group, ThrottleConfig *throttle_cfg,

    BlockdevDetectZeroesOptions *detect_zeroes, Error **errp)

{

    const char *discard;

    Error *local_error = NULL;

    const char *aio;



    if (bdrv_flags) {

        if (!qemu_opt_get_bool(opts, "read-only", false)) {

            *bdrv_flags |= BDRV_O_RDWR;

        }

        if (qemu_opt_get_bool(opts, "copy-on-read", false)) {

            *bdrv_flags |= BDRV_O_COPY_ON_READ;

        }



        if ((discard = qemu_opt_get(opts, "discard")) != NULL) {

            if (bdrv_parse_discard_flags(discard, bdrv_flags) != 0) {

                error_setg(errp, "Invalid discard option");

                return;

            }

        }



        if ((aio = qemu_opt_get(opts, "aio")) != NULL) {

            if (!strcmp(aio, "native")) {

                *bdrv_flags |= BDRV_O_NATIVE_AIO;

            } else if (!strcmp(aio, "threads")) {

                /* this is the default */

            } else {

               error_setg(errp, "invalid aio option");

               return;

            }

        }

    }



    /* disk I/O throttling */

    if (throttling_group) {

        *throttling_group = qemu_opt_get(opts, "throttling.group");

    }



    if (throttle_cfg) {

        memset(throttle_cfg, 0, sizeof(*throttle_cfg));

        throttle_cfg->buckets[THROTTLE_BPS_TOTAL].avg =

            qemu_opt_get_number(opts, "throttling.bps-total", 0);

        throttle_cfg->buckets[THROTTLE_BPS_READ].avg  =

            qemu_opt_get_number(opts, "throttling.bps-read", 0);

        throttle_cfg->buckets[THROTTLE_BPS_WRITE].avg =

            qemu_opt_get_number(opts, "throttling.bps-write", 0);

        throttle_cfg->buckets[THROTTLE_OPS_TOTAL].avg =

            qemu_opt_get_number(opts, "throttling.iops-total", 0);

        throttle_cfg->buckets[THROTTLE_OPS_READ].avg =

            qemu_opt_get_number(opts, "throttling.iops-read", 0);

        throttle_cfg->buckets[THROTTLE_OPS_WRITE].avg =

            qemu_opt_get_number(opts, "throttling.iops-write", 0);



        throttle_cfg->buckets[THROTTLE_BPS_TOTAL].max =

            qemu_opt_get_number(opts, "throttling.bps-total-max", 0);

        throttle_cfg->buckets[THROTTLE_BPS_READ].max  =

            qemu_opt_get_number(opts, "throttling.bps-read-max", 0);

        throttle_cfg->buckets[THROTTLE_BPS_WRITE].max =

            qemu_opt_get_number(opts, "throttling.bps-write-max", 0);

        throttle_cfg->buckets[THROTTLE_OPS_TOTAL].max =

            qemu_opt_get_number(opts, "throttling.iops-total-max", 0);

        throttle_cfg->buckets[THROTTLE_OPS_READ].max =

            qemu_opt_get_number(opts, "throttling.iops-read-max", 0);

        throttle_cfg->buckets[THROTTLE_OPS_WRITE].max =

            qemu_opt_get_number(opts, "throttling.iops-write-max", 0);



        throttle_cfg->op_size =

            qemu_opt_get_number(opts, "throttling.iops-size", 0);



        if (!check_throttle_config(throttle_cfg, errp)) {

            return;

        }

    }



    if (detect_zeroes) {

        *detect_zeroes =

            qapi_enum_parse(BlockdevDetectZeroesOptions_lookup,

                            qemu_opt_get(opts, "detect-zeroes"),

                            BLOCKDEV_DETECT_ZEROES_OPTIONS__MAX,

                            BLOCKDEV_DETECT_ZEROES_OPTIONS_OFF,

                            &local_error);

        if (local_error) {

            error_propagate(errp, local_error);

            return;

        }



        if (bdrv_flags &&

            *detect_zeroes == BLOCKDEV_DETECT_ZEROES_OPTIONS_UNMAP &&

            !(*bdrv_flags & BDRV_O_UNMAP))

        {

            error_setg(errp, "setting detect-zeroes to unmap is not allowed "

                             "without setting discard operation to unmap");

            return;

        }

    }

}
