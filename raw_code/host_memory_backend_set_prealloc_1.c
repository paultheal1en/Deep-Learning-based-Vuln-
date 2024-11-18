static void host_memory_backend_set_prealloc(Object *obj, bool value,

                                             Error **errp)

{

    Error *local_err = NULL;

    HostMemoryBackend *backend = MEMORY_BACKEND(obj);



    if (backend->force_prealloc) {

        if (value) {

            error_setg(errp,

                       "remove -mem-prealloc to use the prealloc property");

            return;

        }

    }



    if (!memory_region_size(&backend->mr)) {

        backend->prealloc = value;

        return;

    }



    if (value && !backend->prealloc) {

        int fd = memory_region_get_fd(&backend->mr);

        void *ptr = memory_region_get_ram_ptr(&backend->mr);

        uint64_t sz = memory_region_size(&backend->mr);



        os_mem_prealloc(fd, ptr, sz, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

        backend->prealloc = true;

    }

}
