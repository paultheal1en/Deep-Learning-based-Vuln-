static GArray *nvdimm_build_device_structure(void)

{

    GSList *device_list = nvdimm_get_plugged_device_list();

    GArray *structures = g_array_new(false, true /* clear */, 1);



    for (; device_list; device_list = device_list->next) {

        DeviceState *dev = device_list->data;



        /* build System Physical Address Range Structure. */

        nvdimm_build_structure_spa(structures, dev);



        /*

         * build Memory Device to System Physical Address Range Mapping

         * Structure.

         */

        nvdimm_build_structure_memdev(structures, dev);



        /* build NVDIMM Control Region Structure. */

        nvdimm_build_structure_dcr(structures, dev);

    }

    g_slist_free(device_list);



    return structures;

}
