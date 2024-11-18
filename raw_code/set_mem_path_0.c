static void set_mem_path(Object *o, const char *str, Error **errp)

{

    HostMemoryBackend *backend = MEMORY_BACKEND(o);

    HostMemoryBackendFile *fb = MEMORY_BACKEND_FILE(o);



    if (memory_region_size(&backend->mr)) {

        error_setg(errp, "cannot change property value");

        return;

    }

    if (fb->mem_path) {

        g_free(fb->mem_path);

    }

    fb->mem_path = g_strdup(str);

}
