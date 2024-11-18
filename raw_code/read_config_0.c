static int read_config(BDRVBlkdebugState *s, const char *filename,

                       QDict *options, Error **errp)

{

    FILE *f = NULL;

    int ret;

    struct add_rule_data d;

    Error *local_err = NULL;



    if (filename) {

        f = fopen(filename, "r");

        if (f == NULL) {

            error_setg_errno(errp, errno, "Could not read blkdebug config file");

            return -errno;

        }



        ret = qemu_config_parse(f, config_groups, filename);

        if (ret < 0) {

            error_setg(errp, "Could not parse blkdebug config file");

            ret = -EINVAL;

            goto fail;

        }

    }



    qemu_config_parse_qdict(options, config_groups, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;

    }



    d.s = s;

    d.action = ACTION_INJECT_ERROR;

    qemu_opts_foreach(&inject_error_opts, add_rule, &d, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;

    }



    d.action = ACTION_SET_STATE;

    qemu_opts_foreach(&set_state_opts, add_rule, &d, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;

    }



    ret = 0;

fail:

    qemu_opts_reset(&inject_error_opts);

    qemu_opts_reset(&set_state_opts);

    if (f) {

        fclose(f);

    }

    return ret;

}
