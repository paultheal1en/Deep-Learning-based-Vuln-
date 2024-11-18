static int qemu_rbd_create(const char *filename, QEMUOptionParameter *options)

{

    int64_t bytes = 0;

    int64_t objsize;

    int obj_order = 0;

    char pool[RBD_MAX_POOL_NAME_SIZE];

    char name[RBD_MAX_IMAGE_NAME_SIZE];

    char snap_buf[RBD_MAX_SNAP_NAME_SIZE];

    char conf[RBD_MAX_CONF_SIZE];

    rados_t cluster;

    rados_ioctx_t io_ctx;

    int ret;



    if (qemu_rbd_parsename(filename, pool, sizeof(pool),

                           snap_buf, sizeof(snap_buf),

                           name, sizeof(name),

                           conf, sizeof(conf)) < 0) {

        return -EINVAL;

    }



    /* Read out options */

    while (options && options->name) {

        if (!strcmp(options->name, BLOCK_OPT_SIZE)) {

            bytes = options->value.n;

        } else if (!strcmp(options->name, BLOCK_OPT_CLUSTER_SIZE)) {

            if (options->value.n) {

                objsize = options->value.n;

                if ((objsize - 1) & objsize) {    /* not a power of 2? */

                    error_report("obj size needs to be power of 2");

                    return -EINVAL;

                }

                if (objsize < 4096) {

                    error_report("obj size too small");

                    return -EINVAL;

                }

                obj_order = ffs(objsize) - 1;

            }

        }

        options++;

    }



    if (rados_create(&cluster, NULL) < 0) {

        error_report("error initializing");

        return -EIO;

    }



    if (strstr(conf, "conf=") == NULL) {

        if (rados_conf_read_file(cluster, NULL) < 0) {

            error_report("error reading config file");

            rados_shutdown(cluster);

            return -EIO;

        }

    }



    if (conf[0] != '\0' &&

        qemu_rbd_set_conf(cluster, conf) < 0) {

        error_report("error setting config options");

        rados_shutdown(cluster);

        return -EIO;

    }



    if (rados_connect(cluster) < 0) {

        error_report("error connecting");

        rados_shutdown(cluster);

        return -EIO;

    }



    if (rados_ioctx_create(cluster, pool, &io_ctx) < 0) {

        error_report("error opening pool %s", pool);

        rados_shutdown(cluster);

        return -EIO;

    }



    ret = rbd_create(io_ctx, name, bytes, &obj_order);

    rados_ioctx_destroy(io_ctx);

    rados_shutdown(cluster);



    return ret;

}
