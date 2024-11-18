static int kvmppc_read_host_property(const char *node_path, const char *prop,

                                     void *val, size_t len)

{

    char *path;

    FILE *f;

    int ret;

    int pathlen;



    pathlen = snprintf(NULL, 0, "%s/%s/%s", PROC_DEVTREE_PATH, node_path, prop)

              + 1;

    path = qemu_malloc(pathlen);

    if (path == NULL) {

        ret = -ENOMEM;

        goto out;

    }



    snprintf(path, pathlen, "%s/%s/%s", PROC_DEVTREE_PATH, node_path, prop);



    f = fopen(path, "rb");

    if (f == NULL) {

        ret = errno;

        goto free;

    }



    len = fread(val, len, 1, f);

    if (len != 1) {

        ret = ferror(f);

        goto close;

    }



close:

    fclose(f);

free:

    free(path);

out:

    return ret;

}
