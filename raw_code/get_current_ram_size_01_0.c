static ram_addr_t get_current_ram_size(void)

{

    GSList *list = NULL, *item;

    ram_addr_t size = ram_size;



    pc_dimm_build_list(qdev_get_machine(), &list);

    for (item = list; item; item = g_slist_next(item)) {

        Object *obj = OBJECT(item->data);

        size += object_property_get_int(obj, PC_DIMM_SIZE_PROP, &error_abort);

    }

    g_slist_free(list);



    return size;

}
