static int qemu_rbd_set_keypairs(rados_t cluster, const char *keypairs,

                                 Error **errp)

{

    char *p, *buf;

    char *name;

    char *value;

    Error *local_err = NULL;

    int ret = 0;



    buf = g_strdup(keypairs);

    p = buf;



    while (p) {

        name = qemu_rbd_next_tok(RBD_MAX_CONF_NAME_SIZE, p,

                                 '=', "conf option name", &p, &local_err);

        if (local_err) {

            break;

        }



        if (!p) {

            error_setg(errp, "conf option %s has no value", name);

            ret = -EINVAL;

            break;

        }



        value = qemu_rbd_next_tok(RBD_MAX_CONF_VAL_SIZE, p,

                                  ':', "conf option value", &p, &local_err);

        if (local_err) {

            break;

        }



        ret = rados_conf_set(cluster, name, value);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "invalid conf option %s", name);

            ret = -EINVAL;

            break;

        }

    }



    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

    }

    g_free(buf);

    return ret;

}
