static void load_symbols(struct elfhdr *hdr, int fd)

{

    unsigned int i, nsyms;

    struct elf_shdr sechdr, symtab, strtab;

    char *strings;

    struct syminfo *s;

    struct elf_sym *syms;



    lseek(fd, hdr->e_shoff, SEEK_SET);

    for (i = 0; i < hdr->e_shnum; i++) {

        if (read(fd, &sechdr, sizeof(sechdr)) != sizeof(sechdr))

            return;

#ifdef BSWAP_NEEDED

        bswap_shdr(&sechdr);

#endif

        if (sechdr.sh_type == SHT_SYMTAB) {

            symtab = sechdr;

            lseek(fd, hdr->e_shoff

                  + sizeof(sechdr) * sechdr.sh_link, SEEK_SET);

            if (read(fd, &strtab, sizeof(strtab))

                != sizeof(strtab))

                return;

#ifdef BSWAP_NEEDED

            bswap_shdr(&strtab);

#endif

            goto found;

        }

    }

    return; /* Shouldn't happen... */



 found:

    /* Now know where the strtab and symtab are.  Snarf them. */

    s = malloc(sizeof(*s));

    syms = malloc(symtab.sh_size);

    if (!syms)

        return;

    s->disas_strtab = strings = malloc(strtab.sh_size);

    if (!s->disas_strtab)

        return;



    lseek(fd, symtab.sh_offset, SEEK_SET);

    if (read(fd, syms, symtab.sh_size) != symtab.sh_size)

        return;



    nsyms = symtab.sh_size / sizeof(struct elf_sym);



    i = 0;

    while (i < nsyms) {

#ifdef BSWAP_NEEDED

        bswap_sym(syms + i);

#endif

        // Throw away entries which we do not need.

        if (syms[i].st_shndx == SHN_UNDEF ||

                syms[i].st_shndx >= SHN_LORESERVE ||

                ELF_ST_TYPE(syms[i].st_info) != STT_FUNC) {

            nsyms--;

            if (i < nsyms) {

                syms[i] = syms[nsyms];

            }

            continue;

        }

#if defined(TARGET_ARM) || defined (TARGET_MIPS)

        /* The bottom address bit marks a Thumb or MIPS16 symbol.  */

        syms[i].st_value &= ~(target_ulong)1;

#endif

        i++;

    }

    syms = realloc(syms, nsyms * sizeof(*syms));



    qsort(syms, nsyms, sizeof(*syms), symcmp);



    lseek(fd, strtab.sh_offset, SEEK_SET);

    if (read(fd, strings, strtab.sh_size) != strtab.sh_size)

        return;

    s->disas_num_syms = nsyms;

#if ELF_CLASS == ELFCLASS32

    s->disas_symtab.elf32 = syms;

    s->lookup_symbol = (lookup_symbol_t)lookup_symbolxx;

#else

    s->disas_symtab.elf64 = syms;

    s->lookup_symbol = (lookup_symbol_t)lookup_symbolxx;

#endif

    s->next = syminfos;

    syminfos = s;

}
