static void fill_elf_note_phdr(struct elf_phdr *phdr, int sz, off_t offset)

{

    phdr->p_type = PT_NOTE;

    phdr->p_offset = offset;

    phdr->p_vaddr = 0;

    phdr->p_paddr = 0;

    phdr->p_filesz = sz;

    phdr->p_memsz = 0;

    phdr->p_flags = 0;

    phdr->p_align = 0;



#ifdef BSWAP_NEEDED

    bswap_phdr(phdr);

#endif

}
