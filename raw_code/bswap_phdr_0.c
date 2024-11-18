static void bswap_phdr(struct elf_phdr *phdr)

{

    bswap32s(&phdr->p_type);            /* Segment type */

    bswaptls(&phdr->p_offset);          /* Segment file offset */

    bswaptls(&phdr->p_vaddr);           /* Segment virtual address */

    bswaptls(&phdr->p_paddr);           /* Segment physical address */

    bswaptls(&phdr->p_filesz);          /* Segment size in file */

    bswaptls(&phdr->p_memsz);           /* Segment size in memory */

    bswap32s(&phdr->p_flags);           /* Segment flags */

    bswaptls(&phdr->p_align);           /* Segment alignment */

}
