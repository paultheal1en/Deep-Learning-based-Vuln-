static int read_config(BDRVBlkdebugState *s, const char *filename, Error **errp)

{

    FILE *f;

    int ret;

    struct add_rule_data d;



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



    d.s = s;

    d.action = ACTION_INJECT_ERROR;

    qemu_opts_foreach(&inject_error_opts, add_rule, &d, 0);



    d.action = ACTION_SET_STATE;

    qemu_opts_foreach(&set_state_opts, add_rule, &d, 0);



    ret = 0;

fail:

    qemu_opts_reset(&inject_error_opts);

    qemu_opts_reset(&set_state_opts);

    fclose(f);

    return ret;

}
