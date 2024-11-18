static void pc_dimm_get_size(Object *obj, Visitor *v, const char *name,

                             void *opaque, Error **errp)

{

    int64_t value;

    MemoryRegion *mr;

    PCDIMMDevice *dimm = PC_DIMM(obj);



    mr = host_memory_backend_get_memory(dimm->hostmem, errp);

    value = memory_region_size(mr);



    visit_type_int(v, name, &value, errp);

}
