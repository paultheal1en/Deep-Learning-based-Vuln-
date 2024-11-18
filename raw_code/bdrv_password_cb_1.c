static void bdrv_password_cb(void *opaque, const char *password,

                             void *readline_opaque)

{

    Monitor *mon = opaque;

    BlockDriverState *bs = readline_opaque;

    int ret = 0;

    Error *local_err = NULL;



    bdrv_add_key(bs, password, &local_err);

    if (local_err) {

        error_report_err(local_err);

        ret = -EPERM;

    }

    if (mon->password_completion_cb)

        mon->password_completion_cb(mon->password_opaque, ret);



    monitor_read_command(mon, 1);

}
