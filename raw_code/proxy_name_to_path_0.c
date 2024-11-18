static int proxy_name_to_path(FsContext *ctx, V9fsPath *dir_path,

                              const char *name, V9fsPath *target)

{

    if (dir_path) {

        v9fs_string_sprintf((V9fsString *)target, "%s/%s",

                            dir_path->data, name);

    } else {

        v9fs_string_sprintf((V9fsString *)target, "%s", name);

    }

    /* Bump the size for including terminating NULL */

    target->size++;

    return 0;

}
