static int write_elf32_note(DumpState *s)

{

    target_phys_addr_t begin = s->memory_offset - s->note_size;

    Elf32_Phdr phdr;

    int endian = s->dump_info.d_endian;

    int ret;



    memset(&phdr, 0, sizeof(Elf32_Phdr));

    phdr.p_type = cpu_convert_to_target32(PT_NOTE, endian);

    phdr.p_offset = cpu_convert_to_target32(begin, endian);

    phdr.p_paddr = 0;

    phdr.p_filesz = cpu_convert_to_target32(s->note_size, endian);

    phdr.p_memsz = cpu_convert_to_target32(s->note_size, endian);

    phdr.p_vaddr = 0;



    ret = fd_write_vmcore(&phdr, sizeof(Elf32_Phdr), s);

    if (ret < 0) {

        dump_error(s, "dump: failed to write program header table.\n");

        return -1;

    }



    return 0;

}
