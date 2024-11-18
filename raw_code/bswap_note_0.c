static void bswap_note(struct elf_note *en)

{

    bswap32s(&en->n_namesz);

    bswap32s(&en->n_descsz);

    bswap32s(&en->n_type);

}
