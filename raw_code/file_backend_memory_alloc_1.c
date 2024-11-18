file_backend_memory_alloc(HostMemoryBackend *backend, Error **errp)

{

    HostMemoryBackendFile *fb = MEMORY_BACKEND_FILE(backend);



    if (!backend->size) {

        error_setg(errp, "can't create backend with size 0");

        return;

    }

    if (!fb->mem_path) {

        error_setg(errp, "mem-path property not set");

        return;

    }

#ifndef CONFIG_LINUX

    error_setg(errp, "-mem-path not supported on this host");

#else

    if (!memory_region_size(&backend->mr)) {

        backend->force_prealloc = mem_prealloc;

        memory_region_init_ram_from_file(&backend->mr, OBJECT(backend),

                                 object_get_canonical_path(OBJECT(backend)),

                                 backend->size, fb->share,

                                 fb->mem_path, errp);

    }

#endif

}
