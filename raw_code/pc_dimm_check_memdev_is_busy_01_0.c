static void pc_dimm_check_memdev_is_busy(const Object *obj, const char *name,

                                      Object *val, Error **errp)

{

    Error *local_err = NULL;



    if (host_memory_backend_is_mapped(MEMORY_BACKEND(val))) {

        char *path = object_get_canonical_path_component(val);

        error_setg(&local_err, "can't use already busy memdev: %s", path);

        g_free(path);

    } else {

        qdev_prop_allow_set_link_before_realize(obj, name, val, &local_err);

    }



    error_propagate(errp, local_err);

}
