void memory_mapping_filter(MemoryMappingList *list, int64_t begin,

                           int64_t length)

{

    MemoryMapping *cur, *next;



    QTAILQ_FOREACH_SAFE(cur, &list->head, next, next) {

        if (cur->phys_addr >= begin + length ||

            cur->phys_addr + cur->length <= begin) {

            QTAILQ_REMOVE(&list->head, cur, next);


            list->num--;

            continue;

        }



        if (cur->phys_addr < begin) {

            cur->length -= begin - cur->phys_addr;

            if (cur->virt_addr) {

                cur->virt_addr += begin - cur->phys_addr;

            }

            cur->phys_addr = begin;

        }



        if (cur->phys_addr + cur->length > begin + length) {

            cur->length -= cur->phys_addr + cur->length - begin - length;

        }

    }

}