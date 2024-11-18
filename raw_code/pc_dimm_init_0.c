static void pc_dimm_init(Object *obj)

{

    PCDIMMDevice *dimm = PC_DIMM(obj);



    object_property_add(obj, PC_DIMM_SIZE_PROP, "int", pc_dimm_get_size,

                        NULL, NULL, NULL, &error_abort);

    object_property_add_link(obj, PC_DIMM_MEMDEV_PROP, TYPE_MEMORY_BACKEND,

                             (Object **)&dimm->hostmem,

                             qdev_prop_allow_set_link_before_realize,

                             OBJ_PROP_LINK_UNREF_ON_RELEASE,

                             &error_abort);

}
