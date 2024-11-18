ram_addr_t get_current_ram_size(void)

{

    MemoryDeviceInfoList *info_list = NULL;

    MemoryDeviceInfoList **prev = &info_list;

    MemoryDeviceInfoList *info;

    ram_addr_t size = ram_size;



    qmp_pc_dimm_device_list(qdev_get_machine(), &prev);

    for (info = info_list; info; info = info->next) {

        MemoryDeviceInfo *value = info->value;



        if (value) {

            switch (value->kind) {

            case MEMORY_DEVICE_INFO_KIND_DIMM:

                size += value->dimm->size;

                break;

            default:

                break;

            }

        }

    }

    qapi_free_MemoryDeviceInfoList(info_list);



    return size;

}
