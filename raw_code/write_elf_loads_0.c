static int write_elf_loads(DumpState *s)

{

    hwaddr offset;

    MemoryMapping *memory_mapping;

    uint32_t phdr_index = 1;

    int ret;

    uint32_t max_index;



    if (s->have_section) {

        max_index = s->sh_info;

    } else {

        max_index = s->phdr_num;

    }



    QTAILQ_FOREACH(memory_mapping, &s->list.head, next) {

        offset = get_offset(memory_mapping->phys_addr, s);

        if (s->dump_info.d_class == ELFCLASS64) {

            ret = write_elf64_load(s, memory_mapping, phdr_index++, offset);

        } else {

            ret = write_elf32_load(s, memory_mapping, phdr_index++, offset);

        }



        if (ret < 0) {

            return -1;

        }



        if (phdr_index >= max_index) {

            break;

        }

    }



    return 0;

}
