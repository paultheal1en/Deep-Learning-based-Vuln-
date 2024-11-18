static void fill_elf_header(struct elfhdr *elf, int segs, uint16_t machine,

                            uint32_t flags)

{

    (void) memset(elf, 0, sizeof(*elf));



    (void) memcpy(elf->e_ident, ELFMAG, SELFMAG);

    elf->e_ident[EI_CLASS] = ELF_CLASS;

    elf->e_ident[EI_DATA] = ELF_DATA;

    elf->e_ident[EI_VERSION] = EV_CURRENT;

    elf->e_ident[EI_OSABI] = ELF_OSABI;



    elf->e_type = ET_CORE;

    elf->e_machine = machine;

    elf->e_version = EV_CURRENT;

    elf->e_phoff = sizeof(struct elfhdr);

    elf->e_flags = flags;

    elf->e_ehsize = sizeof(struct elfhdr);

    elf->e_phentsize = sizeof(struct elf_phdr);

    elf->e_phnum = segs;



#ifdef BSWAP_NEEDED

    bswap_ehdr(elf);

#endif

}
