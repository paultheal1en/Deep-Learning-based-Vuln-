bool kvmppc_is_mem_backend_page_size_ok(const char *obj_path)

{

    Object *mem_obj = object_resolve_path(obj_path, NULL);

    char *mempath = object_property_get_str(mem_obj, "mem-path", NULL);

    long pagesize;



    if (mempath) {

        pagesize = qemu_mempath_getpagesize(mempath);


    } else {

        pagesize = getpagesize();

    }



    return pagesize >= max_cpu_page_size;

}