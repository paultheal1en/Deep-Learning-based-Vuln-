static bool elf_check_ehdr(struct elfhdr *ehdr)

{

    return (elf_check_arch(ehdr->e_machine)

            && ehdr->e_ehsize == sizeof(struct elfhdr)

            && ehdr->e_phentsize == sizeof(struct elf_phdr)

            && ehdr->e_shentsize == sizeof(struct elf_shdr)

            && (ehdr->e_type == ET_EXEC || ehdr->e_type == ET_DYN));

}
