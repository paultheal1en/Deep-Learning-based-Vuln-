int load_elf_as(const char *filename,

                uint64_t (*translate_fn)(void *, uint64_t),

                void *translate_opaque, uint64_t *pentry, uint64_t *lowaddr,

                uint64_t *highaddr, int big_endian, int elf_machine,

                int clear_lsb, int data_swab, AddressSpace *as)

{

    int fd, data_order, target_data_order, must_swab, ret = ELF_LOAD_FAILED;

    uint8_t e_ident[EI_NIDENT];



    fd = open(filename, O_RDONLY | O_BINARY);

    if (fd < 0) {

        perror(filename);

        return -1;

    }

    if (read(fd, e_ident, sizeof(e_ident)) != sizeof(e_ident))

        goto fail;

    if (e_ident[0] != ELFMAG0 ||

        e_ident[1] != ELFMAG1 ||

        e_ident[2] != ELFMAG2 ||

        e_ident[3] != ELFMAG3) {

        ret = ELF_LOAD_NOT_ELF;

        goto fail;

    }

#ifdef HOST_WORDS_BIGENDIAN

    data_order = ELFDATA2MSB;

#else

    data_order = ELFDATA2LSB;

#endif

    must_swab = data_order != e_ident[EI_DATA];

    if (big_endian) {

        target_data_order = ELFDATA2MSB;

    } else {

        target_data_order = ELFDATA2LSB;

    }



    if (target_data_order != e_ident[EI_DATA]) {

        ret = ELF_LOAD_WRONG_ENDIAN;

        goto fail;

    }



    lseek(fd, 0, SEEK_SET);

    if (e_ident[EI_CLASS] == ELFCLASS64) {

        ret = load_elf64(filename, fd, translate_fn, translate_opaque, must_swab,

                         pentry, lowaddr, highaddr, elf_machine, clear_lsb,

                         data_swab, as);

    } else {

        ret = load_elf32(filename, fd, translate_fn, translate_opaque, must_swab,

                         pentry, lowaddr, highaddr, elf_machine, clear_lsb,

                         data_swab, as);

    }



 fail:

    close(fd);

    return ret;

}
